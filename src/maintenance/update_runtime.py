"""Update only an installed maintenance stack's image and matching Lambda package."""

from __future__ import annotations

import argparse
import json
import subprocess


def aws(service, operation, arguments):
    result = subprocess.run(
        ["aws", service, operation, "--cli-input-json", json.dumps(arguments)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout or "{}")


def update(stack_name, image, source_sha, *, call=aws):
    stacks = call("cloudformation", "list-stacks", {})["StackSummaries"]
    if not any(
        s["StackName"] == stack_name and s["StackStatus"] != "DELETE_COMPLETE" for s in stacks
    ):
        return {"updated": False, "reason": "stack has not been installed"}
    stack = call("cloudformation", "describe-stacks", {"StackName": stack_name})["Stacks"][0]
    parameters = {p["ParameterKey"]: p["ParameterValue"] for p in stack["Parameters"]}
    service = call(
        "ecs",
        "describe-services",
        {"cluster": parameters["ClusterName"], "services": [parameters["ServiceName"]]},
    )["services"][0]
    definition = call(
        "ecs", "describe-task-definition", {"taskDefinition": service["taskDefinition"]}
    )["taskDefinition"]
    live_image = next(
        c["image"] for c in definition["containerDefinitions"] if c["name"] == "fantasy-predictor"
    )
    if live_image.rsplit(":", 1)[-1] != source_sha:
        return {"updated": False, "reason": "a different source is deployed"}
    replacements = {
        "WorkerImage": image,
        "LambdaCodeKey": f"maintenance/images/{source_sha}/control.zip",
    }
    if all(parameters[k] == v for k, v in replacements.items()):
        return {"updated": False, "reason": "runtime already matches"}
    call(
        "cloudformation",
        "update-stack",
        {
            "StackName": stack_name,
            "UsePreviousTemplate": True,
            "Capabilities": ["CAPABILITY_IAM"],
            "Parameters": [
                {"ParameterKey": key, "ParameterValue": replacements[key]}
                if key in replacements
                else {"ParameterKey": key, "UsePreviousValue": True}
                for key in parameters
            ],
        },
    )
    return {"updated": True, "source_sha": source_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    result = update(args.stack, args.image, args.source_sha)
    print(json.dumps(result), flush=True)
    if result["updated"]:
        subprocess.run(
            ["aws", "cloudformation", "wait", "stack-update-complete", "--stack-name", args.stack],
            check=True,
        )


if __name__ == "__main__":
    main()
