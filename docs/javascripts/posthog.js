/* PostHog analytics for the Tessera docs site (GitHub Pages, static: no proxy). */
(function () {
  var POSTHOG_KEY = "phc_oLm8fb7L4CXDqGdJcc7a6boCtR8rVjf6dNx7G995eGdy";
  var APP_SLUG = "tessera";

  // Skip local `mkdocs serve` unless explicitly enabled with localStorage.posthog_dev = "1".
  var host = window.location.hostname;
  var isLocal = host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0" || host === "[::1]";
  if (isLocal && window.localStorage.getItem("posthog_dev") !== "1") return;

  // Official PostHog loader snippet (loads array.js from the US assets host).
  !function(t,e){var o,n,p,r;e.__SV||(window.posthog&&window.posthog.__loaded)||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSurveysLoaded onSessionId getSurveys getActiveMatchingSurveys renderSurvey canRenderSurvey canRenderSurveyAsync identify setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException loadToolbar get_property getSessionProperty createPersonProfile opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing get_explicit_consent_status is_capturing clear_opt_in_out_capturing debug getPageViewId captureTraceFeedback captureTraceMetric".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);

  posthog.init(POSTHOG_KEY, {
    api_host: "https://us.i.posthog.com",
    ui_host: "https://us.posthog.com",
    defaults: "2026-08-30",
    capture_pageview: "history_change",
    capture_pageleave: true,
    autocapture: true,
    enable_heatmaps: true,
    capture_dead_clicks: true,
    capture_exceptions: true,
    capture_performance: { web_vitals: true },
    session_recording: { maskAllInputs: true }
  });
  posthog.register({ app: APP_SLUG });

  function page() {
    return window.location.pathname;
  }

  function setup() {
    document.addEventListener("click", function (ev) {
      var target = ev.target;
      if (!(target instanceof Element)) return;

      // Code block copied (Material's content.code.copy button; markup differs across versions).
      var copyBtn = target.closest(".md-clipboard, [data-md-type='copy']");
      if (copyBtn) {
        var block = copyBtn.closest("[class*='language-']") || copyBtn.closest(".highlight");
        var cls = (block && block.className) || "";
        var lang = (cls.match(/language-([\w-]+)/) || [])[1] || null;
        posthog.capture("code_snippet_copied", { page: page(), language: lang });
        return;
      }

      var link = target.closest("a[href]");
      if (!link) return;
      var url;
      try { url = new URL(link.getAttribute("href"), window.location.href); } catch (e) { return; }

      if (url.origin !== window.location.origin) {
        posthog.capture("outbound_link_clicked", {
          page: page(),
          url: url.origin + url.pathname,
          is_repo_link: url.hostname === "github.com"
        });
      }
    }, true);

    // Light/dark palette switch.
    document.querySelectorAll("[data-md-component='palette'] input").forEach(function (input) {
      input.addEventListener("change", function () {
        posthog.capture("theme_toggled", { scheme: input.getAttribute("data-md-color-scheme") });
      });
    });

    // Search: one event per settled query (debounced).
    var searchInput = document.querySelector("[data-md-component='search-query']");
    if (searchInput) {
      var timer = null;
      var lastQuery = "";
      searchInput.addEventListener("input", function () {
        clearTimeout(timer);
        timer = setTimeout(function () {
          var q = searchInput.value.trim();
          if (q.length < 2 || q === lastQuery) return;
          lastQuery = q;
          var results = document.querySelectorAll(".md-search-result__item").length;
          posthog.capture("docs_search_run", {
            query: q.slice(0, 100),
            query_length: q.length,
            results_count: results,
            page: page()
          });
        }, 1000);
      });
    }

    // Page read to the end (scrolled past 90% of the document).
    var readSent = false;
    var started = Date.now();
    window.addEventListener("scroll", function () {
      if (readSent) return;
      var doc = document.documentElement;
      var scrollable = doc.scrollHeight - window.innerHeight;
      if (scrollable <= 0) return;
      if (window.scrollY / scrollable >= 0.9) {
        readSent = true;
        posthog.capture("doc_page_read", {
          page: page(),
          title: document.title,
          seconds_on_page: Math.round((Date.now() - started) / 1000)
        });
      }
    }, { passive: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
})();
