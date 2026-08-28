from app.core.database import test_connection


if __name__ == "__main__":
    result = test_connection()

    print("PostgreSQL connection successful!")
    print(f"Database : {result['database']}")
    print(f"User     : {result['user']}")
    print(f"Version  : {result['version']}")