# EtherGraph Queries
## Subject Expert or Who spoke most about specific terms
```graphql
{
  var(func: anyofterms(originalForm, "Knowledge Graph")) {
    originalForm
    ~hasKeywords @groupby(authoredBy){
      a as count(uid)
  	}
  }
  byWords(func: uid(a), orderdesc:val(a)) {
    xid
    name
    val(a)
  }
}
```
**Output**
```json
{
  "data": {
    "byWords": [
      {
        "xid": "716067a6-0a1a-4034-abc4-9a12ecafb39b",
        "val(a)": 132
      },
      {
        "xid": "7e7ccbba-232d-411a-a95a-d3f244a35f40",
        "name": "Shashank",
        "val(a)": 112
      },
      {
        "xid": "b1e8787a-9a1f-4859-ac11-cbb6a8124fd9",
        "name": "Venkata Dikshit",
        "val(a)": 100
      },
      {
        "xid": "8d6db5f7-d9b7-4c54-ba38-fe710ffcaf3f",
        "name": "Krishna Sai",
        "val(a)": 62
      },
      {
        "xid": "62b6ae1d-7f83-4b0b-b205-5f7c72bc3368",
        "name": "Karthik Muralidharan",
        "val(a)": 54
      },
      {
        "xid": "fb52cb66-3aec-4795-aee3-8ccfd904d315",
        "name": "Reagan Rewop",
        "val(a)": 49
      },
      {
        "xid": "81a3e154-6937-4fce-ba1c-f972faa209b2",
        "name": "Arjun Kini",
        "val(a)": 32
      },
      {
        "xid": "3f01f203-2f58-4b17-8faf-de6b437058ae",
        "name": "Venkat Dikshit",
        "val(a)": 31
      },
      {
        "xid": "8fff81b5-b2f1-4aa5-ad67-405f3e8127f3",
        "name": "Krishna Sai",
        "val(a)": 27
      },
      {
        "xid": "2c944512-17a0-4912-9a16-6a3408da807c",
        "name": "Franklin Ferdinand",
        "val(a)": 23
      },
      {
        "xid": "9fa5972a-f3c6-44c7-b34d-fb8848970841",
        "name": "F3 Testing",
        "val(a)": 16
      },
      {
        "xid": "84fbaa66-a247-4ea2-9ae0-53f3a2e519d6",
        "name": "mithun",
        "val(a)": 14
      },
      {
        "xid": "ecfeeb75-7f0a-4d47-af1e-bd513929264a",
        "name": "Shubham",
        "val(a)": 11
      },
      {
        "xid": "4d9cebed-fd22-435a-bf1e-7f971e70fc52",
        "val(a)": 9
      },
      {
        "xid": "05b7302f-97fe-4618-943c-386f6462f8d8",
        "name": "Franklin Ferdinand",
        "val(a)": 6
      },
      {
        "xid": "04913c7d-dddf-4656-837f-78c7b9b08167",
        "name": "F1 Testing",
        "val(a)": 6
      },
      {
        "xid": "b782dae5-06de-4b7a-bc7d-789a1ad005e5",
        "name": "Cullen",
        "val(a)": 5
      },
      {
        "xid": "1075edb1-49b4-4187-a299-e0d7c533681a",
        "val(a)": 3
      },
      {
        "xid": "19d496b4-d782-4dfe-a4b3-e82b7e8740c2",
        "val(a)": 3
      },
      {
        "xid": "70caa626-9d8e-4869-a45f-7ea91ade3472",
        "val(a)": 2
      },
      {
        "xid": "65bb8395-2fb5-4409-a4bb-59bb707f1375",
        "name": "Vani",
        "val(a)": 2
      },
      {
        "xid": "08b4b1cb-18b2-4160-9c2b-adcf7930ca0e",
        "name": "F3 Testing",
        "val(a)": 2
      },
      {
        "xid": "81c8eba6-7ad8-4d80-b0ff-44379778b831",
        "val(a)": 2
      },
      {
        "xid": "c66797a9-2e6d-46ad-9573-926e57f7dac3",
        "name": "Nisha Yadav",
        "val(a)": 2
      },
      {
        "xid": "32ae0ffb-974c-4427-a606-51ff2d68789b",
        "val(a)": 1
      },
      {
        "xid": "700aae73-9c4b-48ad-894f-af018420a483",
        "val(a)": 1
      },
      {
        "xid": "1a215425-8449-4fca-ba95-7d768b595b80",
        "name": "Vamshi Krishna",
        "val(a)": 1
      },
      {
        "xid": "fd7268ca-e163-46cb-8d88-613da95d7dcb",
        "val(a)": 1
      },
      {
        "xid": "692f8c85-0a15-4f04-beb4-80c3413e790e",
        "name": "shashank",
        "val(a)": 1
      },
      {
        "xid": "a25a2bc9-7e48-4382-93a5-011ea2563ba3",
        "val(a)": 1
      },
      {
        "xid": "936e3738-e9ab-4d0e-baa0-7bbdec89b6b0",
        "name": "Cullen Childress",
        "val(a)": 1
      },
      {
        "xid": "75bdf310-110b-4b8f-ab88-b16fafce920e",
        "name": "Trishanth Diwate",
        "val(a)": 1
      },
      {
        "xid": "b6806dc1-0a18-4d82-9c25-68f0c497825d",
        "name": "Karthik Muralidharan",
        "val(a)": 1
      }
    ]
  },
  "extensions": {
    "server_latency": {
      "parsing_ns": 18846,
      "processing_ns": 31372226,
      "encoding_ns": 101133,
      "assign_timestamp_ns": 524782
    },
    "txn": {
      "start_ts": 55618
    }
  }
}
```

## How many **Users** are related to a specific **User**
```graphql
{
  var(func: type("TranscriptionSegment")) {
    authoredBy @groupby(groupedWith){
      a as count(uid)
  	}
  }
  byRelUsers(func: uid(a), orderdesc:val(a)) {
    user_id: xid
    name: name
    num_users: val(a)
    grouped_users: groupedWith {
      name: name
    }
  }
}
```
**Output**
```json
{
  "data": {
    "byRelUsers": [
      {
        "user_id": "8d6db5f7-d9b7-4c54-ba38-fe710ffcaf3f",
        "name": "Krishna Sai",
        "num_users": 5,
        "grouped_users": [
          {
            "name": "tapasya"
          },
          {
            "name": "Venkata Dikshit"
          },
          {
            "name": "mithun"
          },
          {
            "name": "Karthik Muralidharan"
          },
          {
            "name": "Shweta"
          }
        ]
      },
      {
        "user_id": "b1e8787a-9a1f-4859-ac11-cbb6a8124fd9",
        "name": "Venkata Dikshit",
        "num_users": 4,
        "grouped_users": [
          {
            "name": "Shashank"
          },
          {
            "name": "Krishna Sai"
          },
          {
            "name": "Shubham"
          },
          {
            "name": "Reagan Rewop"
          }
        ]
      },
      {
        "user_id": "fb52cb66-3aec-4795-aee3-8ccfd904d315",
        "name": "Reagan Rewop",
        "num_users": 3,
        "grouped_users": [
          {
            "name": "Venkata Dikshit"
          },
          {
            "name": "Shashank"
          },
          {
            "name": "Shubham"
          }
        ]
      },
      {
        "user_id": "7e7ccbba-232d-411a-a95a-d3f244a35f40",
        "name": "Shashank",
        "num_users": 3,
        "grouped_users": [
          {
            "name": "Venkata Dikshit"
          },
          {
            "name": "Karthik Muralidharan"
          },
          {
            "name": "Reagan Rewop"
          }
        ]
      },
      {
        "user_id": "3fb76ac2-2273-4e69-86f4-21762840f972",
        "name": "Shweta",
        "num_users": 3,
        "grouped_users": [
          {
            "name": "tapasya"
          },
          {
            "name": "Krishna Sai"
          },
          {
            "name": "Cullen Childress"
          }
        ]
      },
      {
        "user_id": "62b6ae1d-7f83-4b0b-b205-5f7c72bc3368",
        "name": "Karthik Muralidharan",
        "num_users": 2,
        "grouped_users": [
          {
            "name": "Shashank"
          },
          {
            "name": "Krishna Sai"
          }
        ]
      },
      {
        "user_id": "ecfeeb75-7f0a-4d47-af1e-bd513929264a",
        "name": "Shubham",
        "num_users": 2,
        "grouped_users": [
          {
            "name": "Venkata Dikshit"
          },
          {
            "name": "Reagan Rewop"
          }
        ]
      },
      {
        "user_id": "fcdae948-70c0-4b1a-9232-06fb97cb2176",
        "name": "tapasya",
        "num_users": 2,
        "grouped_users": [
          {
            "name": "Krishna Sai"
          },
          {
            "name": "Shweta"
          }
        ]
      },
      {
        "user_id": "b782dae5-06de-4b7a-bc7d-789a1ad005e5",
        "name": "Cullen",
        "num_users": 1
      },
      {
        "user_id": "8fff81b5-b2f1-4aa5-ad67-405f3e8127f3",
        "name": "Krishna Sai",
        "num_users": 1
      },
      {
        "user_id": "936e3738-e9ab-4d0e-baa0-7bbdec89b6b0",
        "name": "Cullen Childress",
        "num_users": 1,
        "grouped_users": [
          {
            "name": "Shweta"
          }
        ]
      },
      {
        "user_id": "84fbaa66-a247-4ea2-9ae0-53f3a2e519d6",
        "name": "mithun",
        "num_users": 1,
        "grouped_users": [
          {
            "name": "Krishna Sai"
          }
        ]
      },
      {
        "user_id": "15b1fdcd-48e0-466a-8699-6dfa32ad12a9",
        "num_users": 1,
        "grouped_users": [
          {
            "name": "Krishna Sai"
          }
        ]
      },
      {
        "user_id": "e3cd4976-c1d6-4d74-a08a-6f8b4ad73936",
        "num_users": 1,
        "grouped_users": [
          {
            "name": "Cullen"
          }
        ]
      }
    ]
  },
  "extensions": {
    "server_latency": {
      "parsing_ns": 17923,
      "processing_ns": 676143039,
      "encoding_ns": 104595,
      "assign_timestamp_ns": 665940
    },
    "txn": {
      "start_ts": 55619
    }
  }
}
```

## Get summary segments info
```graphql
{
  highlightSegments(func: type("ContextSession")) {
    hasSegment @filter(eq(highlight, true)){
      xid
      text
      highlight
      authoredBy {
        xid
        name
      }
      hasKeywords {
        originalForm
      }
      hasEntities {
        originalForm
      }
    }
  }
}
```
**Output**
```json
{
  "data": {
    "highlightSegments": [
      {
        "hasSegment": [
          {
            "xid": "ca7cb669-ce0a-4849-be7e-65c41a7de613",
            "text": "Hey, I'm Kyle and welcome to text would tv. We're gonna be running a ton of services from this 23 terabyte server in an upcoming video before we get started. We have to go over Docker. Thank you to hakam for sponsoring this episode. You can get 10% off your own custom domain name at hover.com forward slash Tech squid. What is Docker Docker is mainly a software development platform and kind of a virtualization technology that makes it easy for us to develop and deploy apps inside of neatly packaged virtual container. Erised environments meaning apps run the same no matter where they are or what machine they're running on Docker containers can be deployed to just about any machine without any compatibility issues. So your software stay is system agnostic making software simpler to use less work to develop and easier to maintain and deploy these containers running on your computer or server act like little micro heaters each with very specific jobs each with their own operating system their own isolated CPU processes memory and network resources and because of this they can be easily added removed stopped and started. Again, without affecting each other or the host machine containers usually run one specific tasks such as a mySQL database or a node.js application and then their Network together and potentially scaled a developer will usually start by accessing the docker hug and online Cloud repository of Docker containers and pull one containing a pre-configured environment for their specific programming language such as Ruby or node js with all the files and Frameworks needed to get started home users can experience Docker as well using containers for popular apps like Plex Media Server next. Loud and many other open-source apps and tools many of which we will be installing in upcoming videos Docker is a form of virtualization but unlike virtual machines. The resources are shared directly with the host. This allows you to run many Docker containers where you may only be able to run a few virtual machines. You see a virtual machine has to quarantine off a set amount of resources. It's hard drive space memory and processing power emulate hardware and then boot an entire operating system. Then the VM communicates with the host computer via a translator application running on the ",
            "highlight": true,
            "authoredBy": {
              "xid": "716067a6-0a1a-4034-abc4-9a12ecafb39b"
            },
            "hasKeywords": [
              {
                "originalForm": "software stay"
              },
              {
                "originalForm": "Node Js"
              },
              {
                "originalForm": "pre-configured environment"
              },
              {
                "originalForm": "host machine containers"
              },
              {
                "originalForm": "upcoming video"
              },
              {
                "originalForm": "terabyte server"
              },
              {
                "originalForm": "act like little micro heaters"
              },
              {
                "originalForm": "power emulate hardware"
              },
              {
                "originalForm": "few virtual machines"
              },
              {
                "originalForm": "system agnostic making software simpler"
              },
              {
                "originalForm": "software development platform"
              },
              {
                "originalForm": "form of virtualization"
              },
              {
                "originalForm": "host computer"
              },
              {
                "originalForm": "set amount of resources"
              },
              {
                "originalForm": "compatibility issues"
              },
              {
                "originalForm": "ton of services"
              },
              {
                "originalForm": "get started home users"
              },
              {
                "originalForm": "translator application"
              },
              {
                "originalForm": "hard drive space memory"
              },
              {
                "originalForm": "virtualization technology"
              },
              {
                "originalForm": "network resources"
              },
              {
                "originalForm": "files and Frameworks"
              },
              {
                "originalForm": "specific tasks"
              },
              {
                "originalForm": "entire operating system"
              },
              {
                "originalForm": "node.js application"
              },
              {
                "originalForm": "containers for popular apps"
              },
              {
                "originalForm": "own operating system"
              },
              {
                "originalForm": "many other open-source apps"
              },
              {
                "originalForm": "Docker Hug"
              },
              {
                "originalForm": "virtual container"
              },
              {
                "originalForm": "specific jobs"
              },
              {
                "originalForm": "deploy these containers"
              },
              {
                "originalForm": "run many Docker containers"
              },
              {
                "originalForm": "own custom domain name at hover.com forward slash Tech squid"
              },
              {
                "originalForm": "users can experience Docker"
              },
              {
                "originalForm": "videos Docker"
              },
              {
                "originalForm": "Plex Media Server"
              },
              {
                "originalForm": "Erised environments"
              },
              {
                "originalForm": "go over Docker"
              },
              {
                "originalForm": "online Cloud repository of Docker containers"
              },
              {
                "originalForm": "mySQL database"
              }
            ],
            "hasEntities": [
              {
                "originalForm": "Node Js"
              },
              {
                "originalForm": "Online Cloud"
              },
              {
                "originalForm": "Docker Hug"
              },
              {
                "originalForm": "VM"
              },
              {
                "originalForm": "Hakam"
              },
              {
                "originalForm": "Machine"
              },
              {
                "originalForm": "Loud"
              },
              {
                "originalForm": "Ruby"
              },
              {
                "originalForm": "Tech Squid"
              },
              {
                "originalForm": "Docker"
              },
              {
                "originalForm": "Plex Media Server"
              },
              {
                "originalForm": "Mysql"
              },
              {
                "originalForm": "Erised"
              },
              {
                "originalForm": "Virtualization"
              },
              {
                "originalForm": "Kyle"
              },
              {
                "originalForm": "Hover"
              }
            ]
          }
```

## Get summary info (summary highlights, keywords, entities, grouped users)
```graphql
{
  var(func: type(“Context"))@filter(eq(xid, “01E3109FJD9WS5QQ656BJ3VF5H”)) @cascade{
    uid
    attribute
    associatedMind {
      mindId: xid
      attribute
    }
    hasMeeting (orderdesc: startTime, first:10) {
      meetingId: xid
      attribute
      hasSegment {
        belongsTo {
	   sum_id as uid
        }
      }
    }
  }
    summaryInfo(func: uid(sum_id)) {
      segments: ~belongsTo {
      meetings: ~hasSegment {
      meetingId: xid
      }
    }
      summaryId: xid
      users: hasUser {
        userId: xid
        relatedUsers: groupedWith {
          relatedUser: xid
        }
      }
      hasKeywords {
        phrase: originalForm
      }
      hasEntities {
        entities: originalForm
      }
    }
} 
```
