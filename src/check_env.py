import sys

def main():
    print("Python executable:")
    print(sys.executable)
    print()

    print("Checking installed packages...")

    import fastapi
    import uvicorn
    import pandas
    import numpy
    import pydicom
    import PIL

    print("fastapi:", fastapi.__version__)
    print("uvicorn:", uvicorn.__version__)
    print("pandas:", pandas.__version__)
    print("numpy:", numpy.__version__)
    print("pydicom:", pydicom.__version__)
    print("pillow:", PIL.__version__)

    print("\n✅ Environment check PASSED!")

if __name__ == "__main__":
    main()
