# Keymaker Docs

## Running the Docs Site Locally

Make sure you have [Hugo](https://gohugo.io/installation/) installed.

Pull down the [DoIt](https://github.com/HEIGE-PCloud/DoIt) theme which is included as a submodule.

```sh
git submodule update --init --recursive
```

Start the docs site in a local hugo server.
```sh
cd docs
hugo serve --contentDir=content/0.1.0
```
