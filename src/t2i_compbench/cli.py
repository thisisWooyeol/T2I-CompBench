import os
import subprocess
import sys
from importlib import resources


def main():
    """
    Finds and executes the run_t2i_compbench.sh script,
    passing along any command-line arguments.
    """
    try:
        script_path_obj = resources.files("t2i_compbench").joinpath("run_t2i_compbench.sh")

        with resources.as_file(script_path_obj) as script_path:
            if not os.access(script_path, os.X_OK):
                os.chmod(script_path, 0o755)

            args = [str(script_path)] + sys.argv[1:]
            subprocess.run(args, check=True)

    except FileNotFoundError:
        print("Error: run_t2i_compbench.sh not found within the package.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
