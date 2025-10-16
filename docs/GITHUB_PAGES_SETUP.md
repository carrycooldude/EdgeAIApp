# GitHub Pages Configuration for EdgeAI Documentation

This directory contains the documentation website for EdgeAI, hosted on GitHub Pages.

## 🌐 Website URL

The documentation website is available at:
**https://carrycooldude.github.io/EdgeAIApp-ExecuTorch/**

## 📁 Directory Structure

```
docs/
├── index.html                    # Main website homepage
├── styles.css                    # Website styling
├── README.md                     # Documentation overview
├── technical/                    # Technical documentation
│   ├── REAL_EXECUTORCH_QNN_INTEGRATION.md
│   ├── IMPLEMENTATION_ANALYSIS.md
│   └── PROJECT_STRUCTURE.md
├── setup/                        # Setup guides
│   ├── QUALCOMM_AIHUB_SETUP.md
│   ├── EXECUTORCH_SETUP.md
│   └── ANDROID_SETUP.md
└── releases/                     # Release notes
    ├── RELEASE_NOTES_v1.4.0.md
    ├── RELEASE_NOTES_v1.3.0.md
    └── RELEASE_NOTES_v1.2.0.md
```

## 🚀 GitHub Pages Setup

To enable GitHub Pages for this repository:

1. Go to **Settings** → **Pages**
2. Select **Source**: Deploy from a branch
3. Select **Branch**: main
4. Select **Folder**: /docs
5. Click **Save**

## 🔧 Local Development

To preview the website locally:

```bash
# Navigate to docs directory
cd docs

# Start a local server (Python 3)
python -m http.server 8000

# Or use Node.js
npx serve .

# Or use PHP
php -S localhost:8000
```

Then visit: http://localhost:8000

## 📝 Content Guidelines

### Writing Documentation
- Use clear, concise language
- Include code examples where appropriate
- Add screenshots for UI elements
- Keep documentation up to date

### File Naming
- Use descriptive filenames
- Use hyphens instead of spaces
- Include version numbers for releases
- Use lowercase for consistency

### Markdown Formatting
- Use proper heading hierarchy (H1 → H2 → H3)
- Include table of contents for long documents
- Use code blocks with syntax highlighting
- Add links to related documentation

## 🎨 Styling Guidelines

The website uses a custom CSS framework with:
- **Color Scheme**: Blue gradient theme (#667eea to #764ba2)
- **Typography**: System fonts for better performance
- **Layout**: Responsive grid system
- **Components**: Cards, buttons, code blocks, navigation

## 🔄 Updating Documentation

1. **Edit files** in the appropriate directory
2. **Test locally** using a local server
3. **Commit changes** to the main branch
4. **GitHub Pages** will automatically rebuild

## 📊 Analytics

GitHub Pages provides basic analytics:
- Page views
- Referrer information
- Popular content

Access analytics in **Settings** → **Pages** → **Analytics**

## 🛠️ Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file with your domain
2. Configure DNS settings
3. Enable HTTPS in GitHub Pages settings

## 📱 Mobile Optimization

The website is fully responsive and optimized for:
- **Desktop**: Full layout with sidebar navigation
- **Tablet**: Adapted grid layouts
- **Mobile**: Stacked layout with touch-friendly navigation

## 🔍 SEO Optimization

- **Meta tags**: Description, keywords, Open Graph
- **Structured data**: JSON-LD for better search results
- **Sitemap**: Automatic generation by GitHub Pages
- **Performance**: Optimized images and CSS

## 🚀 Deployment

GitHub Pages automatically deploys when:
- Changes are pushed to the main branch
- Documentation files are updated
- Configuration changes are made

Deployment typically takes 1-2 minutes.

## 📞 Support

For documentation website issues:
- Check GitHub Pages status
- Review build logs in Actions
- Test locally before pushing
- Use GitHub Issues for bugs

---

**This documentation website showcases the professional quality of the EdgeAI project! 🌟**
