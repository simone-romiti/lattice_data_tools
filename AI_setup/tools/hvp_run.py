import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

try:
    import numpy as np
    import scipy
    import numba
    import dill
except ImportError as e:
    print(f"Dependency missing: {e}")

class HVPController:
    def __init__(self, fast_mode=False, force=False, config_path="ensembles.yaml"):
        self.fast_mode = fast_mode
        self.force = force
        self.config_path = config_path
        self.state_file = Path("pipeline_state.json")
        self.state = self._load_state()
        
        self.stages = {
            "ingestion": "scripts/read-blinded_data.py",
            "mistuning": "scripts/apply-mistunings_ml.py",
            "bounding": "scripts/bounding-ZeroTail.py",
            "systematics": "scripts/apply-UV.py",
            "extrapolation": "scripts/continuum_limit.py"
        }

    def _load_state(self):
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def check_dependencies(self):
        missing = []
        for dep in ["numpy", "scipy", "numba", "dill"]:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        return missing

    def run_stage(self, stage_name, script_name):
        if not self.force and self.state.get(stage_name, {}).get("status") == "success":
            print(f"Skipping stage {stage_name} (already completed).")
            return True

        print(f"Executing stage: {stage_name} using {script_name}...")
        os.chdir(os.path.dirname(os.path.abspath(controller.config_path)))
        # Change CWD to the config directory so scripts find ensembles.yaml
        os.chdir(os.path.dirname(os.path.abspath(controller.config_path)))
        args = [sys.executable, script_name]
        if self.fast_mode:
            args.append("--fast")
        
        env = os.environ.copy()
        # Ensure the library root and AI_setup/scripts are in PYTHONPATH
        lib_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scripts_path = os.path.join(lib_root, "scripts")
        env["PYTHONPATH"] = f"{lib_root}:{scripts_path}:{env.get('PYTHONPATH', '')}"

        try:
            result = subprocess.run(args, env=env, check=True, capture_output=True, text=True)
            print(result.stdout)
            self.state[stage_name] = {"status": "success", "timestamp": str(Path().stat().st_mtime)}
            self._save_state()
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error during stage {stage_name}: {e.stderr}")
            self.state[stage_name] = {"status": "failed", "error": e.stderr}
            self._save_state()
            return False

    def run_pipeline(self):
        missing_deps = self.check_dependencies()
        if missing_deps:
            print(f"Critical: Missing dependencies: {missing_deps}")
            sys.exit(1)

        for stage_name, script_name in self.stages.items():
            script_path = Path(__file__).parent.parent / script_name
            if not script_path.exists():
                print(f"Warning: Script {script_name} not found at {script_path}. Skipping...")
                continue
                
            success = self.run_stage(stage_name, str(script_path))
            if not success:
                print(f"Pipeline halted at stage {stage_name}.")
                break
        else:
            print("HVP Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HVP Pipeline Controller")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode")
    parser.add_argument("--force", action="store_true", help="Force re-run")
    parser.add_argument("--config", default="ensembles.yaml", help="Path to config")
    args = parser.parse_args()
    controller = HVPController(fast_mode=args.fast, force=args.force, config_path=args.config)
    controller.run_pipeline()
