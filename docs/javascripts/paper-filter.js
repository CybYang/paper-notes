(() => {
  const normalize = (value) =>
    String(value || "")
      .normalize("NFKC")
      .toLocaleLowerCase("zh-CN")
      .trim();

  const extractSummary = (html) => {
    const template = document.createElement("template");
    template.innerHTML = html || "";
    const paragraph = [...template.content.querySelectorAll("p")]
      .map((item) => item.textContent.trim())
      .find(Boolean);
    return paragraph || "打开查看这篇论文阅读笔记。";
  };

  const createElement = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };

  const createPaperCard = (paper) => {
    const card = createElement("a", "paper-result");
    card.href = new URL(`../${paper.location}`, window.location.href).href;

    const eyebrow = createElement("div", "paper-result__eyebrow");
    eyebrow.append(
      createElement("span", "", "Paper note"),
      createElement("span", "paper-result__arrow", "↗")
    );

    const title = createElement("h2", "", paper.title);
    const summary = createElement("p", "paper-result__summary", paper.summary);
    const tags = createElement("div", "paper-result__tags");
    tags.setAttribute("aria-label", "论文标签");
    paper.tags.forEach((tag) => tags.append(createElement("span", "", tag)));

    card.append(eyebrow, title, summary, tags);
    return card;
  };

  const initializePaperFilter = async () => {
    const root = document.querySelector("[data-paper-filter]");
    if (!root || root.dataset.state) return;
    root.dataset.state = "loading";

    const queryInput = root.querySelector("#paper-filter-query");
    const tagSelect = root.querySelector("#paper-filter-tag");
    const resetButton = root.querySelector(".paper-filter__reset");
    const summary = root.querySelector(".paper-filter__summary");
    const results = root.querySelector(".paper-filter__results");

    try {
      const indexUrl = new URL("../search/search_index.json", window.location.href);
      const response = await fetch(indexUrl);
      if (!response.ok) throw new Error(`Search index returned ${response.status}`);

      const index = await response.json();
      const papers = index.docs
        .filter((item) => /^papers\/[^/#]+\/$/.test(item.location))
        .map((item) => ({
          location: item.location,
          title: item.title,
          summary: extractSummary(item.text),
          tags: Array.isArray(item.tags) ? item.tags : [],
        }))
        .map((paper) => ({
          ...paper,
          searchable: normalize(
            [paper.title, paper.summary, ...paper.tags].join(" ")
          ),
        }));

      const tagCounts = new Map();
      papers.forEach((paper) => {
        paper.tags.forEach((tag) => tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1));
      });

      tagSelect.replaceChildren();
      tagSelect.append(new Option(`全部标签（${papers.length}）`, ""));
      [...tagCounts.keys()]
        .sort((a, b) => a.localeCompare(b, "zh-CN", { sensitivity: "base" }))
        .forEach((tag) => {
          tagSelect.append(new Option(`${tag}（${tagCounts.get(tag)}）`, tag));
        });

      const render = () => {
        const query = normalize(queryInput.value);
        const selectedTag = tagSelect.value;
        const filtered = papers.filter((paper) => {
          const matchesQuery = !query || paper.searchable.includes(query);
          const matchesTag = !selectedTag || paper.tags.includes(selectedTag);
          return matchesQuery && matchesTag;
        });

        results.replaceChildren();
        results.setAttribute("aria-busy", "false");
        resetButton.hidden = !query && !selectedTag;

        const count = createElement("strong", "", String(filtered.length));
        const label = document.createTextNode(
          filtered.length === papers.length
            ? ` / 共 ${papers.length} 篇论文`
            : ` 篇符合条件 / 共 ${papers.length} 篇论文`
        );
        summary.replaceChildren(count, label);

        if (filtered.length) {
          filtered.forEach((paper) => results.append(createPaperCard(paper)));
        } else {
          const empty = createElement("div", "paper-filter__empty");
          empty.append(
            createElement("span", "", "No matches"),
            createElement("h2", "", "没有找到符合条件的论文"),
            createElement("p", "", "可以尝试缩短关键词、切换标签，或者重置筛选条件。")
          );
          results.append(empty);
        }
      };

      queryInput.addEventListener("input", render);
      tagSelect.addEventListener("change", render);
      resetButton.addEventListener("click", () => {
        queryInput.value = "";
        tagSelect.value = "";
        render();
        queryInput.focus();
      });

      root.dataset.state = "ready";
      render();
    } catch (error) {
      root.dataset.state = "error";
      results.setAttribute("aria-busy", "false");
      results.replaceChildren();
      summary.textContent = "论文索引载入失败";
      const message = createElement(
        "p",
        "paper-filter__message",
        "请刷新页面重试，或者使用网站顶部的全文搜索。"
      );
      results.append(message);
      console.error("Failed to initialize paper filter:", error);
    }
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(initializePaperFilter);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializePaperFilter);
  } else {
    initializePaperFilter();
  }
})();
