# Output schemas

JSON outputs are self-describing and always include protocol, condition and state identities. Capture Parquet schemas are enforced in `task6.capture.storage`; metric and result row contracts are enforced before aggregation. `complete.json` is the commit marker and SHA-256 manifest for directory payloads.
