workflow "lint" {
    on = "pull_request"
    resolves = "run flake8"
}

action "run flake8" {
    uses = "jonasrk/flake8-action@master"
    secrets = ["GITHUB_TOKEN"]
}
