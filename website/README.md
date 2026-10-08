# NePPO project page

Source for the project page at <https://www.addikala.com/neppo/>. Pushes to `main` that touch
`website/` are built and deployed to GitHub Pages by `.github/workflows/pages.yml`.

The page content is [`src/paper.mdx`](./src/paper.mdx); figures are in `src/assets/`.

```bash
npm install     # Node 24 or later
npm run dev     # live preview at http://localhost:4321
npm run build   # static site in ./dist
```

Built from Roman Hauksson-Neill's
[academic project page template](https://github.com/RomanHauksson/academic-project-astro-template),
which is licensed under
[CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/). See
[`documentation.md`](./documentation.md) for the template's components and options.
