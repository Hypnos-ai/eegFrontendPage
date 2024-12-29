import os
import subprocess
import sys
import src.main
def run_vbs_script():
    # Path to the VBScript file
    vbs_path = os.path.join(os.path.dirname(sys.argv[0]), "run_hidden.vbs")
    
    if not os.path.exists(vbs_path):
        print(f"Error: VBScript file not found at {vbs_path}")
        sys.exit(1)
    
    try:
        # Run the VBScript using wscript.exe
        subprocess.run(["wscript.exe", vbs_path], check=True)
        print("VBScript executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while executing VBScript: {e}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")

if __name__ == "__main__":
    run_vbs_script()
    #main.main()
    
