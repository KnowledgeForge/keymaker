---
categories: ["docs"]
weight: 50
title: Contributing
tags: ["how-to", "community"]
series: ["contributing"]
series_weight: 10
lightgallery: true
---

Keymaker is an open source project and there is lots of room and tons of ways to contribute.

# Contributing Code

Currently the project does not have a rigid set of contribution rules but here are some tips.

- The CI that runs for PRs uses the commands `make lint` and `make test`. Once those are passing locally, PR checks should also pass.
- If you want early feedback on a branch before it's ready, it can be useful to open a PR as a draft and link to it in an open issue.
- When it makes sense, try to pair your PR with a new or existing issue. This can help track multiple connected PRs.
- If you have an idea for big changes or changes that touch many parts of the codebase, feel free to first open an issue requesting
a community sync meeting to have a casual discussion. Not only is it helpful to brainstorm together but others may also be eager
to jump in and help!

# Contributing Docs

To run the Keymaker docs site locally, Make sure you have [Hugo](https://gohugo.io/installation/) installed.

Pull down the [DoIt](https://github.com/HEIGE-PCloud/DoIt) theme which is included as a submodule.
```sh
git submodule update --init --recursive
```

Change into the docs directory, then start up the hugo server.
```sh
cd docs
hugo server --contentDir=content/0.1.0
```

The docs site is designed in a way to make contributing new content as easy as possible. All markdown files can be found in
a flat directory located at `docs/content/<latest version>`. Instead of directory based organization or an explicitly configured navbar, the
documentation site takes advantage of the `tags`, `categories`, and `series` front-matter at the top of each file to determine placement. Try
not to think too hard about which to set for each as we occasionally use a language model to determine the right values given the content of the
page. In summary, add a markdown file to the latest docs content directory and things should pretty much just work!