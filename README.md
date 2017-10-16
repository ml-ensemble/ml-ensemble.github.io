<div align="center">
<img src="docs/source/_static/img/logo.png" width="50%"><br><br>
</div>
<hr>

Repository hosting the website of [ML-ensemble](http://ml-ensemble.com). 

The website can be viewed by opening any ``html`` page, such as ``index.html`` in the root directory. 
To build locally, you need [npm](https://www.npmjs.com/get-npm) and [gulp](https://www.npmjs.com/package/gulp) and [Sphinx](http://www.sphinx-doc.org/en/stable).

### Setup for building locally

1. Install npm, gulp and Sphinx with your favorite package manager.

2. Clone repo ``git clone https://github.com/ml-ensemble/ml-ensemble.github.io ml-ensemble-web``

3. Install dependencies: ``cd ml-ensemble-web; npm install``. 

### Build

To regenerate all site content, run
```bash
bash build.sh
```

To serve locally with live reloads, use

```
gulp dev
```

To statically rebuild the landing page, use ``gulp``, and to rebuild the documentation navigate to ``docs/source`` and execute ``make populate``. To clear documentation use ``make clean-cache``. 
Note that this requires you to run ``make populate`` again to populate the full website. 

### Development workflow

To make changes to the website, follow the below procedure.

1. If your changes are the **docs** directory, i.e. the API documentation or tutorials, you first need to make your changes to the ``mlens`` repo. Use ``make docs`` in the ``mlens/docs/source`` directory to view your changes. Commit and push when ready.

2. Make your changes (or copy them from step 1 to ``docs/source``). Use ``gulp dev`` to view your changes in real-time (changes to docs needs to be re-loaded with ``make populate``).

2. Run ``build bash.sh`` to build full site. Ensure no errors where triggered and the website builds correctly. Watch out for unnecessary build directories and files. If encountered, update ``.gitignore`` to avoid committing them.

4. Commit changes, push to remote and create a PR ticket.


## License

MIT License

Copyright (c) 2017 Sebastian Flennerhag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
