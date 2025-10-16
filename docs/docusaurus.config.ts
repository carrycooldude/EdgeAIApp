import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'EdgeAI Documentation',
  tagline: 'Real ExecuTorch + QNN Integration for Llama3.2-1B inference on Android',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://carrycooldude.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/EdgeAIApp-ExecuTorch/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'carrycooldude', // Usually your GitHub org/user name.
  projectName: 'EdgeAIApp-ExecuTorch', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/tree/main/docs/',
        },
               blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/edgeai-social-card.svg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'EdgeAI',
      logo: {
        alt: 'EdgeAI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://github.com/carrycooldude/EdgeAIApp-ExecuTorch',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/intro',
            },
            {
              label: 'Technical Docs',
              to: '/docs/technical/project-structure',
            },
            {
              label: 'Setup Guides',
              to: '/docs/setup/qualcomm-aihub-setup',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/carrycooldude/EdgeAIApp-ExecuTorch',
            },
            {
              label: 'Issues',
              href: 'https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/issues',
            },
            {
              label: 'Discussions',
              href: 'https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/discussions',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Release Notes',
              to: '/docs/releases/release-notes-v1-4-0',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} EdgeAI. Built with EdgeAI Documentation Platform.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
