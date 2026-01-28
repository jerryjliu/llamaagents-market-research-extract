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
from llama_cloud.types.file_query_params import Filter
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


class ExtractionState(BaseModel):
    file_id: str | None = None
    file_path: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    parse_job_id: str | None = None
    extract_job_id: str | None = None


class ProcessFileWorkflow(Workflow):
    """Extract key findings from market research reports with visual understanding."""

    @step()
    async def start_parsing(
        self,
        event: FileEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        parse_config: Annotated[
            ParseConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="parse",
                label="Parsing Settings",
                description="High-quality parsing for visual content understanding",
            ),
        ],
    ) -> ParseJobStartedEvent:
        """Download document and start high-quality parsing for visual understanding."""
        file_id = event.file_id
        logger.info(f"Processing research report: {file_id}")

        # Download file from cloud storage
        files = await llama_cloud_client.files.query(filter=Filter(file_ids=[file_id]))
        file_metadata = files.items[0]
        file_url = await llama_cloud_client.files.get(file_id=file_id)

        temp_dir = tempfile.gettempdir()
        filename = file_metadata.name
        file_path = os.path.join(temp_dir, filename)

        async with httpx.AsyncClient() as client:
            async with client.stream("GET", file_url.url) as response:
                with open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

        file_content = Path(file_path).read_bytes()

        ctx.write_event_to_stream(
            Status(level="info", message=f"Parsing visuals in {filename}")
        )

        # Start parsing with agentic tier for visual understanding
        parse_job = await llama_cloud_client.parsing.create(
            tier=parse_config.tier,
            version=parse_config.version,
            file_id=file_id,
        )

        async with ctx.store.edit_state() as state:
            state.file_id = file_id
            state.file_path = file_path
            state.filename = filename
            state.file_hash = hashlib.sha256(file_content).hexdigest()
            state.parse_job_id = parse_job.id

        return ParseJobStartedEvent()

    @step()
    async def start_extraction(
        self,
        event: ParseJobStartedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Extraction Settings",
                description="Schema and settings for extracting market research findings",
            ),
        ],
    ) -> ExtractJobStartedEvent:
        """Wait for parsing to complete, then start extraction of key findings."""
        state = await ctx.store.get_state()
        if state.parse_job_id is None:
            raise ValueError("Parse job ID cannot be null")

        # Wait for parsing to complete
        await llama_cloud_client.parsing.wait_for_completion(state.parse_job_id)

        ctx.write_event_to_stream(
            Status(level="info", message=f"Extracting findings from {state.filename}")
        )

        # Start extraction with multimodal mode for visual understanding
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
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Extraction Settings",
                description="Schema and settings for extracting market research findings",
            ),
        ],
    ) -> StopEvent:
        """Validate extraction results and save key findings for review."""
        state = await ctx.store.get_state()
        if state.extract_job_id is None:
            raise ValueError("Extract job ID cannot be null")

        await llama_cloud_client.extraction.jobs.wait_for_completion(
            state.extract_job_id
        )

        extracted_result = await llama_cloud_client.extraction.jobs.get_result(
            state.extract_job_id
        )
        extract_run = await llama_cloud_client.extraction.runs.get(
            run_id=extracted_result.run_id
        )

        extracted_event: ExtractedEvent | ExtractedInvalidEvent
        try:
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
            logger.warning(f"Extraction validation issue: {e}")
            extracted_event = ExtractedInvalidEvent(data=e.invalid_item)

        ctx.write_event_to_stream(extracted_event)

        # Save extracted findings for review
        extracted_data = extracted_event.data
        data_dict = extracted_data.model_dump()

        # Remove past data when reprocessing the same file
        if extracted_data.file_hash is not None:
            await llama_cloud_client.beta.agent_data.delete_by_query(
                deployment_name=agent_name or "_public",
                collection=EXTRACTED_DATA_COLLECTION,
                filter={"file_hash": {"eq": extracted_data.file_hash}},
            )

        item = await llama_cloud_client.beta.agent_data.agent_data(
            data=data_dict,
            deployment_name=agent_name or "_public",
            collection=EXTRACTED_DATA_COLLECTION,
        )

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Key findings extracted from {extracted_data.file_name or ''}",
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
