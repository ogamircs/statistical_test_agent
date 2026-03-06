# Processing Loader Design

**Date:** 2026-03-06

## Goal

Replace the visible `Processing...` placeholder with a small animated loading indicator that feels closer to a modern chat UI.

## Problem

The current upload and analysis flow surfaces a literal `Processing...` message. It is functional, but visually flat and distracting relative to the rest of the custom interface.

## Chosen Approach

Enhance processing placeholders in the custom browser layer instead of changing the backend message format.

- Detect assistant articles whose visible text is `Processing...`
- Replace that text with a compact loading row
- Use Wikimedia Commons' public-domain `Ajax-loader.gif` as the remote animated asset
- Fall back to a small CSS spinner if the image fails to load

## Why This Approach

- It works with the current Chainlit message structure
- It centralizes the loader behavior in one place
- It can improve any future `Processing...` placeholder without changing multiple Python call sites

## Expected Behavior

- `Processing...` text is replaced by a small animated loader
- The remote GIF is used by default
- If the remote asset fails, the UI shows a CSS spinner instead of broken content
- The rest of the message content remains unchanged

## Asset Source

- Direct asset: `https://upload.wikimedia.org/wikipedia/commons/d/de/Ajax-loader.gif`
- Source page: Wikimedia Commons `Ajax-loader.gif`
