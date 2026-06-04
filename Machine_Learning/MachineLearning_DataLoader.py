import pandas as pd
from WBCD_Original_Models import WBCD_Original_Analysis, Cross_Validate_Original
from WDBC_Diagnostic_Models import WDBC_Diagnostic_Analysis, Cross_Validate_Diagnostic

def Load_Data(filepath, cols):
    print("=== Loading dataset ===")
    try:
        dataset = pd.read_excel(filepath, usecols = cols)
        print("Loaded successfully")
        return dataset
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()


def main():
    print("=== Machine Learning Data Loader ===")
    print("===  WBCD Original  ===")
    # WBCD -> Wisconsin Breast Cancer Dataset Original
    WBCD_Original = Load_Data("Machine_Learning/DataSets/WBCD(Original).xlsx", "B:K")
    WBCD_Original_Analysis(WBCD_Original.copy())
    Cross_Validate_Original(WBCD_Original.copy())

    print("="*50)
    print("=== WDBC Diagnostic ===")
    # WDBC -> Wisconsin Diagnostic Breast Cancer Dataset
    WDBC_Diagnostic = Load_Data("Machine_Learning/DataSets/WBCD(Diagnostic).xlsx", "B:AF")
    WDBC_Diagnostic_Analysis(WDBC_Diagnostic.copy())
    Cross_Validate_Diagnostic(WDBC_Diagnostic.copy())


if __name__ == "__main__":
    main()

