"""Expiring cross-run publication leases, also callable from AWS-CLI-only CI jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from contextlib import contextmanager


class LeaseBusy(RuntimeError):
    pass


class AwsCliDynamo:
    def __getattr__(self, name):
        def invoke(**arguments):
            result = subprocess.run(
                [
                    "aws",
                    "dynamodb",
                    name.replace("_", "-"),
                    "--cli-input-json",
                    json.dumps(arguments),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                if "ConditionalCheckFailedException" in result.stderr:
                    raise LeaseBusy("Lease is held by another publisher")
                raise RuntimeError(result.stderr.strip())
            return json.loads(result.stdout or "{}")

        return invoke


def acquire(client, table: str, scope: str, owner: str, *, seconds=3600, clock=time.time):
    now = int(clock())
    try:
        client.put_item(
            TableName=table,
            Item={"pk": {"S": scope}, "owner": {"S": owner}, "expires": {"N": str(now + seconds)}},
            ConditionExpression="attribute_not_exists(pk) OR #expires < :now",
            ExpressionAttributeNames={"#expires": "expires"},
            ExpressionAttributeValues={":now": {"N": str(now)}},
        )
    except Exception as error:
        if (
            isinstance(error, LeaseBusy)
            or getattr(error, "response", {}).get("Error", {}).get("Code")
            == "ConditionalCheckFailedException"
        ):
            raise LeaseBusy(f"Maintenance lease busy: {scope}") from error
        raise


def assert_owned(client, table: str, scope: str, owner: str, *, clock=time.time):
    item = client.get_item(TableName=table, Key={"pk": {"S": scope}}, ConsistentRead=True).get(
        "Item", {}
    )
    if (
        item.get("owner", {}).get("S") != owner
        or int(item.get("expires", {}).get("N", "0")) <= clock()
    ):
        raise LeaseBusy(f"Maintenance lease expired or changed: {scope}")


def release(client, table: str, scope: str, owner: str):
    try:
        client.delete_item(
            TableName=table,
            Key={"pk": {"S": scope}},
            ConditionExpression="#owner = :owner",
            ExpressionAttributeNames={"#owner": "owner"},
            ExpressionAttributeValues={":owner": {"S": owner}},
        )
    except Exception as error:
        if (
            isinstance(error, LeaseBusy)
            or getattr(error, "response", {}).get("Error", {}).get("Code")
            == "ConditionalCheckFailedException"
        ):
            return  # Never remove a successor's lease.
        raise


@contextmanager
def lease(client, table: str, scope: str, *, seconds=300):
    owner = uuid.uuid4().hex
    acquire(client, table, scope, owner, seconds=seconds)
    try:
        yield lambda: assert_owned(client, table, scope, owner)
    finally:
        release(client, table, scope, owner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("acquire", "release"))
    parser.add_argument("--scope", default="serving-publication")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--wait-seconds", type=int, default=1800)
    parser.add_argument("--lease-seconds", type=int, default=3600)
    args = parser.parse_args()
    table = os.environ.get("FF_MAINTENANCE_LOCK_TABLE", "")
    if not table:
        print("Maintenance coordination is not enabled")
        return
    client = AwsCliDynamo()
    if args.operation == "release":
        release(client, table, args.scope, args.owner)
        return
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            acquire(client, table, args.scope, args.owner, seconds=args.lease_seconds)
            return
        except LeaseBusy:
            if time.monotonic() >= deadline:
                raise
            time.sleep(15)


if __name__ == "__main__":
    main()
