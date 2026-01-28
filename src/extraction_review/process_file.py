import asyncio
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData, InvalidExtractionData
from pydantic import BaseModel
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .clients import agent_name, get_llama_cloud_client, project_id
from .config import (
    EXTRACTED_DATA_COLLECTION,
    ExtractConfig,
    ParseConfig,
    get_extraction_schema,
)

logger = logging.getLogger(__name__)


class FileEvent(StartEvent):
    file_id: str


class Status(Event):
    level: Literal["info", "warning", "error"]
    message: str


class ParseJobStartedEvent(Event):
    pass


class ExtractJobStartedEvent(Event):
    pass


class ExtractedEvent(Event):
    data: ExtractedData


class ExtractedInvalidEvent(Event):
    """Event for extraction results that failed validation."""

    data: ExtractedData[dict[str, Any]]


class ProcessingState(BaseModel):
    file_id: str | None = None
    file_path: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    parse_job_id: str | None = None
    extract_job_id: str | None = None


class ProcessFileWorkflow(Workflow):
    """Parse and extract key findings from market research reports."""

    @step()
    async def start_parsing(
        self,
        event: FileEvent,
        ctx: Context[ProcessingState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        parse_config: Annotated[
            ParseConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="parse",
                label="Document Parsing",
                description="High-quality parsing settings for charts and visuals",
            ),
        ],
    ) -> ParseJobStartedEvent:
        """Download document and start high-quality parsing for visuals."""
        file_id = event.file_id
        logger.info(f"Processing file {file_id}")

        # Download file from cloud storage
        files = await llama_cloud_client.files.list(file_ids=[file_id])
        file_metadata = files.items[0]
        file_url = await llama_cloud_client.files.get(file_id=file_id)

        temp_dir = tempfile.gettempdir()
        filename = file_metadata.name
        file_path = os.path.join(temp_dir, filename)

        logger.info(f"Downloading file {file_url.url} to {file_path}")
        ctx.write_event_to_stream(
            Status(level="info", message=f"Downloading {filename}")
        )

        async with httpx.AsyncClient() as client:
            async with client.stream("GET", file_url.url) as response:
                with open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

        # Compute file hash for deduplication
        file_content = Path(file_path).read_bytes()
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Start parsing with specialized chart parsing for visuals
        ctx.write_event_to_stream(
            Status(level="info", message=f"Parsing {filename} for charts and visuals")
        )

        processing_options = {}
        if parse_config.processing_options.specialized_chart_parsing:
            processing_options["specialized_chart_parsing"] = (
                parse_config.processing_options.specialized_chart_parsing
            )

        parse_job = await llama_cloud_client.parsing.create(
            tier=parse_config.tier,
            version=parse_config.version,
            file_id=file_id,
            processing_options=processing_options if processing_options else None,
        )

        async with ctx.store.edit_state() as state:
            state.file_id = file_id
            state.file_path = file_path
            state.filename = filename
            state.file_hash = file_hash
            state.parse_job_id = parse_job.id

        return ParseJobStartedEvent()

    @step()
    async def start_extraction(
        self,
        event: ParseJobStartedEvent,
        ctx: Context[ProcessingState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Key Findings Extraction",
                description="Schema for extracting key findings from market research",
            ),
        ],
    ) -> ExtractJobStartedEvent:
        """Wait for parsing to complete, then start extraction."""
        state = await ctx.store.get_state()

        # Wait for parsing to complete
        await llama_cloud_client.parsing.wait_for_completion(state.parse_job_id)

        logger.info(f"Parsing complete for {state.filename}")
        ctx.write_event_to_stream(
            Status(level="info", message=f"Extracting key findings from {state.filename}")
        )

        # Start extraction job
        extract_job = await llama_cloud_client.extraction.run(
            config=extract_config.settings.model_dump(),
            data_schema=extract_config.json_schema,
            file_id=state.file_id,
            project_id=project_id,
        )

        async with ctx.store.edit_state() as state:
            state.extract_job_id = extract_job.id

        return ExtractJobStartedEvent()

    @step()
    async def complete_extraction(
        self,
        event: ExtractJobStartedEvent,
        ctx: Context[ProcessingState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Key Findings Extraction",
                description="Schema for extracting key findings from market research",
            ),
        ],
    ) -> StopEvent:
        """Wait for extraction to complete, validate results, and save."""
        state = await ctx.store.get_state()

        # Wait for extraction job to complete
        await llama_cloud_client.extraction.jobs.wait_for_completion(state.extract_job_id)

        # Get extraction result
        extracted_result = await llama_cloud_client.extraction.jobs.get_result(
            state.extract_job_id
        )
        extract_run = await llama_cloud_client.extraction.runs.get(
            run_id=extracted_result.run_id
        )

        # Validate and parse extraction result
        extracted_event: ExtractedEvent | ExtractedInvalidEvent
        try:
            logger.info(f"Extracted data: {extracted_result}")
            schema_class = get_extraction_schema(extract_config.json_schema)
            data = ExtractedData.from_extraction_result(
                result=extract_run,
                schema=schema_class,
                file_name=state.filename,
                file_id=state.file_id,
                file_hash=state.file_hash,
            )
            extracted_event = ExtractedEvent(data=data)
        except InvalidExtractionData as e:
            logger.error(f"Error validating extracted data: {e}", exc_info=True)
            extracted_event = ExtractedInvalidEvent(data=e.invalid_item)

        ctx.write_event_to_stream(extracted_event)

        # Remove past data when reprocessing the same file
        extracted_data = extracted_event.data
        if extracted_data.file_hash is not None:
            await llama_cloud_client.beta.agent_data.delete_by_query(
                deployment_name=agent_name or "_public",
                collection=EXTRACTED_DATA_COLLECTION,
                filter={"file_hash": {"eq": extracted_data.file_hash}},
            )
            logger.info(
                f"Removed past data for file {extracted_data.file_name} with hash {extracted_data.file_hash}"
            )

        # Save the new data
        data_dict = extracted_data.model_dump()
        item = await llama_cloud_client.beta.agent_data.agent_data(
            data=data_dict,
            deployment_name=agent_name or "_public",
            collection=EXTRACTED_DATA_COLLECTION,
        )

        logger.info(f"Saved extracted findings for {extracted_data.file_name or ''}")
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Extracted key findings from {extracted_data.file_name or ''}",
            )
        )

        return StopEvent(result=item.id)


workflow = ProcessFileWorkflow(timeout=None)

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main():
        file = await get_llama_cloud_client().files.create(
            file=Path("test.pdf").open("rb"),
            purpose="extract",
        )
        await workflow.run(start_event=FileEvent(file_id=file.id))

    asyncio.run(main())
