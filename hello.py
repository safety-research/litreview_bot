import asyncio
import base64
import dotenv
import httpx
import instructor
import os
import pathlib
import pydantic
import re
from anthropic import AsyncAnthropic
from miniperscache import cached_async, DefaultArgHasher
from miniperscache.serializer import Serializer
from rich import print
from tqdm import asyncio as tqdm_asyncio
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    Iterator,
    Literal,
    ParamSpec,
    Sequence,
    TypeVar,
)
import mdutils

import typed_argparse as tap

T = TypeVar("T", bound=pydantic.BaseModel)


class PydanticSerializer(Serializer, Generic[T]):
    def __init__(self, model: type[T]):
        self.model = model

    def serialize(self, value: Any) -> bytes:
        return value.model_dump_json().encode("utf-8")

    def deserialize(self, value: bytes) -> T:
        return self.model.model_validate_json(value.decode("utf-8"))


class SummarizationError(Exception):
    pass


async def _process_pdf_raw(
    client: instructor.AsyncInstructor,
    question: str,
    pdf_path: pathlib.Path,
    response_class: type[T],
) -> T:
    pdf_data = base64.standard_b64encode(pdf_path.read_bytes()).decode("utf-8")

    message = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        },
                    },
                    {"type": "text", "text": question},
                ],
            }  # type: ignore
        ],
        response_model=response_class,
    )

    return message


@cached_async("search_scholar")
async def search_scholar(query: str, num: int = 30) -> dict:
    base = "https://serpapi.com/search"
    path_params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": os.environ["SERPAPI_API_KEY"],
        "num": num,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(base, params=path_params)
        resp.raise_for_status()
        return resp.json()


class ScholarResult(pydantic.BaseModel):
    title: str
    snippet: str
    pdf_url: str | None


def process_scholar_results(results: Sequence[dict]) -> Iterator[ScholarResult]:
    for result in results:
        pdf_url = None
        if "resources" in result:
            res = result["resources"]
            for r in res:
                if "file_format" in r and r["file_format"].lower() == "pdf":
                    pdf_url = r["link"]
                    break

        yield ScholarResult(
            title=result["title"], snippet=result["snippet"], pdf_url=pdf_url
        )


class FilterResult(pydantic.BaseModel):
    keep: bool = pydantic.Field(description="Whether the paper should be kept.")


@cached_async(
    "claude_filter_snippet",
    arg_hasher=DefaultArgHasher(skip_args=["client"]),
    value_serializer=PydanticSerializer(FilterResult),
)
async def claude_filter_snippet(
    client: instructor.AsyncInstructor, query: str, title: str, snippet: str
) -> FilterResult:
    message = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": f"""\
Your job is to decide if a scientific paper with a given snippet and search result is relevant to a given query. Please use the tool to answer.
             
Query: {query}

Paper title: {title}

Paper snippet: {snippet}
""",
            }
        ],
        response_model=FilterResult,
    )
    assert isinstance(
        message, FilterResult
    ), f"Expected FilterResult, got {type(message)} for {message}"
    return message


P = ParamSpec("P")
R = TypeVar("R")


def mk_do_with_semaphore(
    semaphore: asyncio.Semaphore, fn: Callable[P, Coroutine[Any, Any, R]]
) -> Callable[P, Coroutine[Any, Any, R]]:
    async def do_with_semaphore(*args: P.args, **kwargs: P.kwargs) -> R:
        async with semaphore:
            return await fn(*args, **kwargs)

    return do_with_semaphore


def name_to_path(name: str) -> str:
    # replace all non-alphanumeric characters with underscores
    return re.sub(r"\W+", "_", name) + ".pdf"


T = TypeVar("T")


def assert_not_none(value: T | None) -> T:
    assert value is not None, f"{value} is None"
    return value


# class FullResult(pydantic.BaseModel):
#     search_query: str
#     results: Sequence[ScholarResult]
#     summarization_results: Sequence[SummarizationResult]


class SummarizationResult(pydantic.BaseModel):
    general_summary: str = pydantic.Field(
        description="A general summary of the paper, in a few sentences."
    )
    method_summary: str = pydantic.Field(
        description="A summary of the method used in the paper."
    )
    score_summary: str = pydantic.Field(
        description="A summary of the results achieved in the paper. Is the method good at some specific subset of domains?"
    )


def write_results(
    file_name: str,
    search_query: str,
    filtering_query: str,
    results: Sequence[ScholarResult],
    summarization_results: Sequence[SummarizationResult],
):
    md = mdutils.MdUtils(file_name=file_name)
    md.new_header(level=1, title="Lit search results")
    md.new_header(level=2, title=f"Query: '{search_query}'")
    md.new_header(level=2, title=f"Filtering query: '{filtering_query}'")
    md.new_header(level=2, title=f"Num results: {len(results)}")
    md.new_line("\n---\n")

    md.new_header(level=2, title="Result table of contents:")

    for r in results:
        md.new_line(f"### [{r.title}]")
        md.new_line(f"{r.snippet}")
        md.new_line(f"[PDF]({r.pdf_url})")
        md.new_line("\n---\n")

    md.new_header(level=2, title="Summarization results")
    md.new_line()

    for r, s in zip(results, summarization_results):
        md.new_header(level=3, title=r.title)

        url_base = "https://www.google.com/search"
        params = {"q": r.title}
        url = httpx.URL(url_base, params=params)
        md.new_line(f"[Google link]({url})")

        md.new_line()

        md.new_header(level=4, title="General summary")
        md.new_line(s.general_summary)
        md.new_line()
        md.new_header(level=4, title="Method summary")
        md.new_line(s.method_summary)
        md.new_line()
        md.new_header(level=4, title="Score summary")
        md.new_line(s.score_summary)
        md.new_line()
        md.new_line("\n---\n")

    return md


class CliArgs(tap.TypedArgs):
    query: str = tap.arg(help="The query to search for.")
    num_results: int = tap.arg(help="The number of results to search for.", default=30)
    output_md_file: str = tap.arg(
        help="The file to write the results to.", default="out.md"
    )
    output_html_file: str = tap.arg(
        help="The file to write the results to.", default="out.html"
    )
    filter_query: str | None = tap.arg(
        help="The query to filter the results by.", default=None
    )


def main(args: CliArgs):
    dotenv.load_dotenv()
    anth = AsyncAnthropic()
    instr_anth = instructor.from_anthropic(anth)

    async def main():
        res = await search_scholar(args.query)
        search_results = list(process_scholar_results(res["organic_results"]))

        print(f"Num results fetched: {len(search_results)}")

        with_pdf_urls = [r for r in search_results if r.pdf_url is not None]
        print(f"Num with PDFs: {len(with_pdf_urls)}")

        sem = asyncio.Semaphore(5)
        filter_fn = mk_do_with_semaphore(sem, claude_filter_snippet)

        filter_query = (
            args.filter_query
            or f"This paper fits the search string '{args.query}' and is relevant."
        )
        filter_answers: list[FilterResult] = await tqdm_asyncio.tqdm.gather(
            *[
                filter_fn(instr_anth, filter_query, r.title, r.snippet)
                for r in with_pdf_urls
            ]
        )

        passed_filter = [r for (f, r) in zip(filter_answers, with_pdf_urls) if f.keep]

        print(f"Num passed filter: {len(passed_filter)}")

        async def fetch_pdf(url: str, name: str):
            dest = pathlib.Path("pdfs") / name_to_path(name)
            if not dest.parent.exists():
                dest.parent.mkdir(parents=True)
            if dest.exists():
                return dest
            async with httpx.AsyncClient() as client:
                resp = await client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            return dest

        fetch_pdf_sem = mk_do_with_semaphore(sem, fetch_pdf)

        pdf_paths: list[pathlib.Path] = await tqdm_asyncio.tqdm.gather(
            *[fetch_pdf_sem(assert_not_none(r.pdf_url), r.title) for r in passed_filter]
        )

        # now, use Claude to summarize
        summarization_sem = asyncio.Semaphore(10)


        # class SummarizationOutcome(pydantic.BaseModel):

        class SummarizationResultWrapper(pydantic.BaseModel):
            res: SummarizationResult
            typ: Literal["success"] = "success"

        class SummarizationErrorWrapper(pydantic.BaseModel):
            res: str
            typ: Literal["error"] = "error"

        class SummarizationOutcome(pydantic.BaseModel):
            res: SummarizationResultWrapper | SummarizationErrorWrapper = (
                pydantic.Field(discriminator="typ")
            )

        @cached_async(
            "process_pdf",
            arg_hasher=DefaultArgHasher(skip_args=["client"]),
            value_serializer=PydanticSerializer(SummarizationOutcome),
        )
        async def process_pdf(
            client: instructor.AsyncInstructor, question: str, pdf_path: pathlib.Path
        ) -> SummarizationOutcome:
            async with summarization_sem:
                try:
                    res = await _process_pdf_raw(
                        client, question, pdf_path, SummarizationResult
                    )
                    return SummarizationOutcome(res=SummarizationResultWrapper(res=res))
                except Exception as e:
                    return SummarizationOutcome(
                        res=SummarizationErrorWrapper(
                            res=f"Error summarizing {pdf_path}: {e}"
                        )
                    )

        summarization_results: list[
            SummarizationOutcome
        ] = await tqdm_asyncio.tqdm.gather(
            *[
                process_pdf(
                    instr_anth,
                    "Please read the given paper and summarize it. Extract the relevant info and return a SummarizationResult.",
                    pdf_path,
                )
                for pdf_path in pdf_paths
            ]
        )

        for doc, r in zip(passed_filter, summarization_results):
            if r.res.typ == "error":
                print("")
                print(f"Error summarizing {doc.title}: {r.res.res}")

        successful_summarization_results = [
            (doc, r.res.res)
            for (doc, r) in zip(passed_filter, summarization_results)
            if r.res.typ == "success"
        ]
        successful_docs, successful_summaries = zip(*successful_summarization_results)

        write_results(
            args.output_md_file,
            args.query,
            filter_query,
            successful_docs,
            successful_summaries,
        ).create_md_file()
        md_contents = pathlib.Path(args.output_md_file).read_text()
        # convert to html
        import pypandoc

        html = pypandoc.convert_text(md_contents, "html", format="markdown")
        with open(args.output_html_file, "w") as f:
            f.write(html)

        print()
        print(
            f"Summarization complete. Got {len(successful_summarization_results)} results. Results written to {args.output_md_file} and {args.output_html_file}"
        )

    asyncio.run(main())


if __name__ == "__main__":
    tap.Parser(CliArgs).bind(main).run()
