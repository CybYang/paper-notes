(() => {
  const scriptUrl = document.currentScript && document.currentScript.src;
  const taxonomyUrl = new URL(
    "../assets/tag-taxonomy.json",
    scriptUrl || window.location.href
  );
  const categoryByTag = new Map();
  const orderByTag = new Map();
  let groups = [];

  const categoryFor = (tag) => categoryByTag.get(String(tag || "").trim()) || "";

  const annotate = (element, tag) => {
    const category = categoryFor(tag);
    if (category) {
      element.dataset.tagCategory = category;
    } else {
      delete element.dataset.tagCategory;
    }
  };

  const compareTags = (left, right) => {
    const leftTag = left.textContent.trim();
    const rightTag = right.textContent.trim();
    const leftOrder = orderByTag.get(leftTag);
    const rightOrder = orderByTag.get(rightTag);

    if (leftOrder !== undefined && rightOrder !== undefined) {
      return leftOrder - rightOrder;
    }
    if (leftOrder !== undefined) return -1;
    if (rightOrder !== undefined) return 1;
    return leftTag.localeCompare(rightTag, "zh-CN", { sensitivity: "base" });
  };

  const sortTags = (root = document) => {
    root
      .querySelectorAll(".md-tags, .paper-result__tags")
      .forEach((container) => {
        [...container.children]
          .sort(compareTags)
          .forEach((tag) => container.append(tag));
      });
  };

  const apply = (root = document) => {
    root
      .querySelectorAll(".md-tag, .paper-result__tags span")
      .forEach((element) => annotate(element, element.textContent));
    sortTags(root);
  };

  const ready = fetch(taxonomyUrl)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Tag taxonomy returned ${response.status}`);
      }
      return response.json();
    })
    .then((taxonomy) => {
      groups = Array.isArray(taxonomy.groups) ? taxonomy.groups : [];
      let order = 0;
      groups.forEach((group) => {
        (group.tags || []).forEach((tag) => {
          categoryByTag.set(tag, group.id);
          orderByTag.set(tag, order++);
        });
      });
      apply(document);
      return groups;
    })
    .catch((error) => {
      console.error("Failed to load tag taxonomy:", error);
      return [];
    });

  window.paperTagTaxonomy = {
    ready,
    categoryFor,
    annotate,
    apply,
    sortTags,
    getGroups: () => groups,
  };

  const initialize = () => ready.then(() => apply(document));

  if (typeof document$ !== "undefined") {
    document$.subscribe(initialize);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();
