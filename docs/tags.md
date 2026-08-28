---
title: 论文库
description: 按论文名称和研究标签筛选阅读笔记。
hide:
  - toc
---

<div class="page-heading">
  <span class="paper-kicker">Paper Explorer</span>
  <h1>查找论文</h1>
  <p>搜索论文名称或内容关键词，也可以选择研究标签，快速找到对应的阅读笔记。</p>
</div>

<div class="paper-filter" data-paper-filter>
  <div class="paper-filter__controls">
    <label class="paper-filter__field paper-filter__search" for="paper-filter-query">
      <span>搜索论文</span>
      <span class="paper-filter__input-wrap">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m21 20-5.2-5.2a7.5 7.5 0 1 0-1 1L20 21zM5.5 10.5a5 5 0 1 1 10 0 5 5 0 0 1-10 0z"/></svg>
        <input id="paper-filter-query" type="search" placeholder="输入论文名称或关键词" autocomplete="off" spellcheck="false">
      </span>
    </label>

    <label class="paper-filter__field" for="paper-filter-tag">
      <span>筛选标签</span>
      <select id="paper-filter-tag">
        <option value="">正在读取标签…</option>
      </select>
    </label>

    <button class="paper-filter__reset" type="button" hidden>重置筛选</button>
  </div>

  <div class="paper-filter__summary" aria-live="polite">
    <span>正在载入论文索引…</span>
  </div>

  <div class="paper-tag-legend" aria-label="标签颜色分类"></div>

  <div class="paper-filter__results" aria-live="polite" aria-busy="true"></div>

  <noscript>
    <p class="paper-filter__message">此筛选页面需要启用 JavaScript。你仍然可以使用网站顶部的全文搜索。</p>
  </noscript>
</div>
