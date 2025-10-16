import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Technical Documentation',
      items: [
        'technical/project-structure',
      ],
    },
    {
      type: 'category',
      label: 'Setup Guides',
      items: [
        'setup/qualcomm-aihub-setup',
      ],
    },
    {
      type: 'category',
      label: 'Release Notes',
      items: [
        'releases/release-notes-v1-4-0',
      ],
    },
  ],
};

export default sidebars;
