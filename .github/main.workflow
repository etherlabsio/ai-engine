workflow "on pull request merge, delete the branch" {
  on = "pull_request"
  resolves = ["branch cleanup"]
}

action "branch cleanup" {
  uses = "jessfraz/branch-cleanup-action@master"
  secrets = ["GITHUB_TOKEN"]

  env = {
    NO_BRANCH_DELETED_EXIT_CODE = "0"
  }
}


workflow "on check suite creation, run flake8 and post results" {
    on = "pull_request"
    resolves = "run flake8"
}

action "run flake8" {
    uses = "tayfun/flake8-your-pr@master"
    secrets = ["GITHUB_TOKEN"]
}
