#!/usr/bin/env bash
curl -H "Content-Type: application/graphql+-" -X POST dgraph-0.dev.internal.etherlabs.io:8080/query -d $'{
          contextUserInfo(func: eq(xid, "01E36HF5E4BM4S0V0JX8C53SZW")) @cascade{
            attribute
            contextId: xid
          hasMeeting(orderdesc: startTime, first: 10) {
            meetingId: xid
            attribute
            startTime: startTime
            associatedMind {
                    xid
                    attribute
                }
            hasSegment {
              attribute
              xid
              text: text
              startTime: startTime
              authoredBy {
                user_id: xid
              }
              hasKeywords {
                value: originalForm
              }
            }
          }
         }
        }'