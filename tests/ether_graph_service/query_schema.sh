#!/usr/bin/env bash
curl -H "Content-Type: application/graphql+-" -X POST dgraph-0.dev.internal.etherlabs.io:8080/query -d $'
         schema {}'