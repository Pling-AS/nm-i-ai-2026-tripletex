from tripletex_agent.config import get_settings
from tripletex_agent.spec_index import TripletexSpecIndex


def test_spec_index_finds_employee_endpoint() -> None:
    index = TripletexSpecIndex(get_settings().tripletex_api_spec_path)
    results = index.search_endpoints("create employee")

    assert any(result["path"] == "/employee" for result in results)


def test_spec_index_ranks_core_product_endpoint_for_plural_query() -> None:
    index = TripletexSpecIndex(get_settings().tripletex_api_spec_path)
    results = index.search_endpoints("get products")

    assert any(result["path"] == "/product" for result in results)


def test_spec_index_loads_customer_schema() -> None:
    index = TripletexSpecIndex(get_settings().tripletex_api_spec_path)
    schema = index.get_schema("Customer")

    assert schema["schema"] == "Customer"
    assert "name" in schema["properties"]


def test_spec_index_gets_exact_employee_endpoint() -> None:
    index = TripletexSpecIndex(get_settings().tripletex_api_spec_path)
    endpoint = index.get_endpoint("GET", "/employee")

    assert endpoint["method"] == "GET"
    assert endpoint["path"] == "/employee"


def test_spec_index_matches_templated_employee_endpoint() -> None:
    index = TripletexSpecIndex(get_settings().tripletex_api_spec_path)
    endpoint = index.get_endpoint("PUT", "/employee/123")

    assert endpoint["method"] == "PUT"
    assert endpoint["path"] == "/employee/{id}"
