# About This Book

![2Sigma School](2sigma-school-logo.svg)

**Python Data Science Handbook** · *Essential tools for working with data*

**2Sigma School** · Alps edition · Updated November 2026

This handbook covers the core of data science in Python: IPython, NumPy, pandas, Matplotlib and Scikit-Learn. This Alps edition runs the book's code in the page, with nothing to install.

| At a glance | |
|---|---|
| **66 pages** | The book's chapters and sections |
| **717 code cells** | Python that runs in the browser, apart from any cells shown for reference |

## How to Use It

Read each page in order and run its code as you go. On a notebook page, run a cell on its own, or choose **Run all cells** in the toolbar to run the page from the top. A cell shown for reference stays on the page, cannot be edited, and is stepped over by Run all. A cell with a blank for you to fill in shows an error until you complete it.

> **Found a problem?** Teachers: choose Report Bug in the left menu, then select the bug icon on the item, or highlight the text. Students: tell your teacher. You can also email [support@alps.2sigma.us](mailto:support@alps.2sigma.us) with the book, chapter and page name.

## Credits

*Python Data Science Handbook* is by Jake VanderPlas.

Source: [https://github.com/jakevdp/PythonDataScienceHandbook](https://github.com/jakevdp/PythonDataScienceHandbook)

License: the text is under [CC BY-NC-ND 3.0](https://creativecommons.org/licenses/by-nc-nd/3.0/) and the code under the [MIT License](https://opensource.org/license/mit).

This edition has been adapted for Alps. As with every book on Alps, its preparation draws on Alps's content-enhancement capabilities, which include the use of language models.

The book's authors are not affiliated with, and do not endorse, this edition.

## Alps Compatibility Changes

This edition differs from its source so that the book works on Alps, where its code runs in the browser rather than on an installed copy of Python. Alps compatibility is the primary difference. Depending on the page, the changes include:

- a table of contents, or changes to one, so that the book imports as chapters and pages
- code adjusted for the library versions that run in the browser, or for libraries and system tools that are not available there
- data read from a place the browser can reach, or replaced where the original can no longer be loaded
- cells shown for reference, when they need something the browser cannot do
- cells marked as expected to fail, such as a blank for the reader to fill in
- corrections to Markdown or code, so that a page renders and runs
- corrections to typos, links and unclear wording in the text and exercises

The numbering of the book's chapters is not affected: this page comes after them. The source above shows the book as its authors published it.

## Datasets That Come with the Book

Several examples use datasets that scikit-learn downloads the first time they run: three from figshare and one from OpenML. In the browser the figshare download can be refused, and the face photos are more than the browser can load. So this edition includes copies of the three figshare datasets in `notebooks/data/sklearn` (32 MB in all), and those pages load them with `bundled_datasets`, which takes the same arguments and returns the same data as scikit-learn's functions.

| Dataset | Page | In this edition |
|---|---|---|
| Olivetti faces, by AT&T Laboratories Cambridge | Customizing Ticks | Included (1.4 MB) |
| 20 Newsgroups, collected by Ken Lang | In Depth: Naive Bayes Classification | Included (15.3 MB) |
| Labeled Faces in the Wild, by the University of Massachusetts Amherst | In Depth: Support Vector Machines, In Depth: Principal Component Analysis, In Depth: Manifold Learning, Application: A Face Detection Pipeline | Included: the faces of everyone with at least 20 photos, cropped and scaled as scikit-learn does (15.0 MB in place of 243 MB). `min_faces_per_person` of 20 or more works; a smaller value, or another size or crop, is not included |
| MNIST handwritten digits, from OpenML | In Depth: Manifold Learning | Not included: shown for reference. The download (about 120 MB) fails in the browser |

The copies were made with scikit-learn's own loaders, and give exactly the data those loaders return. `tools/build_bundled_datasets.py` rebuilds them and checks them against scikit-learn. A cell shown for reference shows its code but does not run, so the results and figures it would produce do not appear on the page.

**Questions?** For any question about this edition, or about how it differs from its source, email [support@alps.2sigma.us](mailto:support@alps.2sigma.us).
