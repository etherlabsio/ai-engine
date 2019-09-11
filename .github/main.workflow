workflow "lint" {
    on = "pull_request"
    resolves = "run flake8"
}

action "run flake8" {
    uses = "tayfun/flake8-your-pr@master"
    secrets = ["GITHUB_TOKEN"]
}
