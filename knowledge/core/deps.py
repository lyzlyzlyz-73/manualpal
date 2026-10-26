from functools import cache

from knowledge.service.file_import_service import ImportFileService
from knowledge.service.query_service import QueryService


@cache
def get_import_file_service():
    return ImportFileService()


@cache
def get_query_service():
    return QueryService()
