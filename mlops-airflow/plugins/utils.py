import re
from mlflow import MlflowClient

def get_next_deployment_experiment_name(base_name="deploy"):
    client = MlflowClient()
    # list_experiments() → search_experiments()
    experiments = client.search_experiments()

    pattern = re.compile(f"^{base_name}-v(\\d+)$")
    max_id = 0
    found = False

    for exp in experiments:
        match = pattern.match(exp.name)
        if match:
            found = True
            num = int(match.group(1))
            max_id = max(max_id, num)

    if not found:
        return f"{base_name}-v1"
    else:
        return f"{base_name}-v{max_id + 1}"
