from pydantic import Field
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

# FastMCP creates an MCP server and registers tools/resources/prompts via decorators.
mcp = FastMCP("DocumentMCP", log_level="ERROR")


# In-memory sample datastore used by the server.
# Keys are document ids, values are document contents.
docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}

@mcp.tool(
    name="Read_docs_contents",
    description="Read the contents of a document and return it as a string"
)
def read_document(
    doc_id: str = Field(description="Id of the document to read")
):
    # MCP tool: callable action exposed to the client.
    # Here, the tool returns raw text content for one document id.
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")

    return docs[doc_id]

@mcp.tool(
    name="edit_document",
    description="Edit a document by replacing a string in the documents contents with a new string"
)
def edit_document(
    docs_id:str = Field(description="Id of the document that will be edited"),
    old_str: str = Field(description="The text to replace. must match exactly, included white space"),
    new_str: str = Field(description="The new text to insert in place of the old test")
):
    # MCP tool: mutates the in-memory doc by doing a simple string replacement.
    # Note: str.replace() does not error if old_str is missing; it just leaves text unchanged.
    if docs_id not in docs:
        raise ValueError(f"Doc with id {docs_id} not found")
    docs[docs_id] = docs[docs_id].replace(old_str, new_str)

@mcp.resource(
    "docs://documents",
    mime_type="application/json"
)
def list_docs() -> list[str]:
    # MCP resource: read-only content addressable by URI.
    # This endpoint returns all available doc ids.
    return list(docs.keys())

@mcp.resource(
    "docs://documents/{docs_id}",
    mime_type="text/plain"
)
def fetch_doc(docs_id: str) -> str:
    # MCP resource endpoint for fetching one document's text by id.
    if docs_id not in docs:
        raise ValueError(f"Doc with id {docs_id} not found")
    return docs[docs_id]


# TODO: Write a prompt to rewrite a doc in markdown format
@mcp.prompt(
    name="format",
    description="Rewrite the contents of a document in Markdown format."
)
def format_document(
    doc_id: str =Field(description="Id of the document to format")
) -> list[base.Message]:
    # MCP prompt: server-provided prompt template that clients/agents can request.
    # Returning UserMessage(s) lets the client inject structured instructions into a chat flow.
    prompt = f"""
    Your goal is to reformat a document to be written with markdown syntax.

    The id of the document you need to reformat is:
    <document_id>
    {doc_id}
    </document_id>

    Add in headers, bullet points, tables, etc as necessary. Feel free to add in structure.
    Use the 'edit_document' tool to edit the document. After the document has been reformatted...
    """
    
    return [base.UserMessage(prompt)]

# TODO: Write a prompt to summarize a doc


if __name__ == "__main__":
    # Starts the MCP server over stdio so local clients can launch/connect as a subprocess.
    mcp.run(transport="stdio")
