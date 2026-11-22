"""Typed objects shared by all modules. No network or UI dependencies."""
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


@dataclass
class Paper:
    uid: str
    title: str
    authors: list[str]
    year: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    url: str = ""
    provider: str = ""


@dataclass
class Document:
    paper: Paper
    text: str
    source_kind: str
    identity_status: str
    warning: str = ""


class Finding(BaseModel):
    """A candidate result, not a verified scientific assertion."""
    variable: str = Field(description="Outcome, variable, or table-row concept")
    statistic: str = Field(description="N, mean, SD, median, %, OR, CI lower, p-value etc.")
    value: str = Field(description="Exact numerical expression from source, e.g. -0.35 or <0.001")
    unit: str = Field(description="Unit or empty string")
    population: str = Field(description="Population/cohort/subgroup")
    table_column: str = Field(description="Source table-column label or empty string")
    source_label: str = Field(description="Exact [Page N], [Section N] or [Table ... Row ...] label")
    evidence_quote: str = Field(description="Exact continuous substring of provided source")


class FindingsBatch(BaseModel):
    findings: list[Finding]


@dataclass
class ExtractedRecord:
    paper_uid: str
    paper_title: str
    doi: str
    finding: Finding
    validation: str
    chunk_index: int

    def to_dict(self):
        f = self.finding
        return {
            "Publication ID": self.paper_uid,
            "Study": self.paper_title,
            "DOI": self.doi,
            "Variable": f.variable,
            "Statistic": f.statistic,
            "Value": f.value,
            "Unit": f.unit,
            "Population": f.population,
            "Table Column": f.table_column,
            "Source Label": f.source_label,
            "Evidence Quote": f.evidence_quote,
            "Validation": self.validation,
            "Chunk": self.chunk_index,
        }
