from auth.security import authenticate_user, seed_default_users
from database.db import get_db, init_db
from rag.admin import create_product, list_products


def test_db_initialization_and_seeding() -> None:
    init_db()
    seed_default_users()

    admin_user = authenticate_user("admin", "admin123")
    assert admin_user is not None
    assert admin_user.role == "ADMIN"

    customer_user = authenticate_user("customer", "customer123")
    assert customer_user is not None
    assert customer_user.role == "CUSTOMER"


def test_product_workspace_creation() -> None:
    init_db()
    prod = create_product("TP-Link", "Archer AX21 Test", "AX21", "V2", "Test router")
    assert prod.id == "archer-ax21-test"

    products = list_products()
    found = any(p.id == "archer-ax21-test" for p in products)
    assert found is True
