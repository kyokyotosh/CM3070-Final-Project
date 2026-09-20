/* ==========================================================================
   interface layer
   --------------------------------------------------------------------------
   This file owns presentation only. It never decides whether a rep was good:
   it renders verdicts that arrive from the backend, in keeping with the
   rule-based analysis layer being the single diagnostic authority.

   main.js talks to it through window.CoachUI:

     CoachUI.setStatus("live" | "connecting" | "offline", label?)
     CoachUI.setExercise("squat")
     CoachUI.setReps(12)
     CoachUI.setCoaching("text")                 // updates the big line only
     CoachUI.pushRep({ rep, exercise, faults, text, metrics, latencyMs })
     CoachUI.reset()
   ========================================================================== */

(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);

    const el = {
        root: document.documentElement,
        themeBtn: $("theme-toggle"),
        themeLabel: document.querySelector(".theme-toggle-label"),
        conn: $("conn-status"),
        connLabel: $("conn-label"),
        exerciseSelect: $("exercise-select"),
        hint: $("viewport-hint"),
        lastBox: $("last-feedback"),
        lastVerdict: $("last-verdict"),
        coaching: $("coaching-text"),
        countBox: document.querySelector(".count"),
        count: $("rep-count"),
        countWord: $("count-word"),
        sessionExercise: $("session-exercise"),
        sessionTime: $("session-time"),
        tallyGood: $("tally-good"),
        tallyFault: $("tally-fault"),
        qualityLatest: $("quality-latest"),
        graphArea: $("graph-area"),
        graphLine: $("graph-line"),
        graphDots: $("graph-dots"),
        graphBand: document.querySelector(".graph-band"),
        graphThreshold: document.querySelector(".graph-threshold"),
        graphCaption: $("graph-caption"),
        feed: $("feedback-feed"),
        feedEmpty: $("feed-empty"),
        clearFeed: $("clear-feed")
    };

    /* ------------------------------------------------------------- theme -- */

    const THEME_KEY = "coach-theme";

    function applyTheme(name) {
        el.root.setAttribute("data-theme", name);
        if (el.themeLabel) el.themeLabel.textContent = name === "dark" ? "Light" : "Dark";
        if (el.themeBtn) {
            el.themeBtn.setAttribute(
                "aria-label",
                name === "dark" ? "Switch to light theme" : "Switch to dark theme"
            );
        }
    }

    let stored = null;
    try {
        stored = localStorage.getItem(THEME_KEY);
    } catch (e) {
        stored = null;
    }

    const prefersLight = window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: light)").matches;

    applyTheme(stored || (prefersLight ? "light" : "dark"));

    if (el.themeBtn) {
        el.themeBtn.addEventListener("click", () => {
            const next = el.root.getAttribute("data-theme") === "dark" ? "light" : "dark";
            applyTheme(next);
            try {
                localStorage.setItem(THEME_KEY, next);
            } catch (e) {
                /* private mode */
            }
        });
    }

    /* --------------------------------------------------------- session -- */

    const state = {
        reps: 0,
        good: 0,
        fault: 0,
        quality: [],
        startedAt: null
    };

    const MAX_POINTS = 24;
    const QUALITY_PASS = 70;

    function pad2(n) {
        return n < 10 ? "0" + n : String(n);
    }

    function tickClock() {
        if (!state.startedAt || !el.sessionTime) return;
        const s = Math.floor((Date.now() - state.startedAt) / 1000);
        el.sessionTime.textContent = Math.floor(s / 60) + ":" + pad2(s % 60);
    }
    setInterval(tickClock, 1000);

    /* ----------------------------------------------------------- labels -- */

    // Backend fault identifiers mapped to words a person training would use.
    const FAULT_LABELS = {
        shallow_depth: "Not deep enough",
        depth: "Not deep enough",
        insufficient_depth: "Not deep enough",
        trunk_lean: "Leaning forward",
        excessive_trunk_lean: "Leaning forward",
        trunk: "Leaning forward",
        knee_travel: "Front knee drifting",
        excessive_knee_travel: "Front knee drifting",
        knee_valgus: "Knees caving in",
        short_stride: "Stride too short"
    };

    // Metric keys mapped to a short label, a unit, and the calibrated threshold
    // they are judged against, so a breached value can be marked in the chip.
    const METRIC_SPECS = {
        knee_angle: { label: "Knee", unit: "\u00B0", digits: 0 },
        min_knee_angle: { label: "Depth", unit: "\u00B0", digits: 0 },
        depth: { label: "Depth", unit: "\u00B0", digits: 0 },
        trunk_angle: { label: "Trunk", unit: "\u00B0", digits: 0 },
        max_trunk_angle: { label: "Trunk", unit: "\u00B0", digits: 0 },
        trunk_lean: { label: "Trunk", unit: "\u00B0", digits: 0 },
        knee_travel: { label: "Knee travel", unit: "", digits: 2 },
        rep_duration: { label: "Duration", unit: "s", digits: 1 },
        tempo: { label: "Tempo", unit: "s", digits: 1 }
    };

    function humanise(key) {
        return String(key).replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
    }

    function faultLabel(fault) {
        if (!fault) return "Needs work";
        if (typeof fault === "object") fault = fault.name || fault.fault || fault.id || "";
        const key = String(fault).toLowerCase();
        return FAULT_LABELS[key] || humanise(key);
    }

    /* ---------------------------------------------------------- quality -- */

    /*  PLACEHOLDER SCORE.
        The backend does not emit a quality score yet. Until it does, the graph
        is driven by this local estimate: a clean rep scores 100, and each fault
        costs a penalty scaled by how far the offending metric sits past its
        calibrated threshold. Thresholds match the values calibrated from real
        reps. Replace this by sending `quality` in the verdict payload. */
    const THRESHOLDS = {
        squat: { depth: 95, trunk: 50, travel: 0.15 },
        lunge: { depth: 110, trunk: 20, travel: 0.15 }
    };

    function estimateQuality(faults, metrics, exercise) {
        if (!faults || faults.length === 0) return 100;

        const t = THRESHOLDS[exercise] || THRESHOLDS.squat;
        const m = metrics || {};
        let score = 100;

        faults.forEach((raw) => {
            const key = String(
                typeof raw === "object" ? (raw.name || raw.fault || "") : raw
            ).toLowerCase();
            let penalty = 24;

            if (key.indexOf("depth") !== -1) {
                const v = m.min_knee_angle ?? m.knee_angle ?? m.depth;
                if (typeof v === "number") {
                    penalty = 14 + Math.min(26, Math.abs(v - t.depth) * 1.1);
                }
            } else if (key.indexOf("trunk") !== -1) {
                const v = m.max_trunk_angle ?? m.trunk_angle ?? m.trunk_lean;
                if (typeof v === "number") {
                    penalty = 14 + Math.min(26, Math.abs(v - t.trunk) * 1.3);
                }
            } else if (key.indexOf("knee") !== -1 || key.indexOf("travel") !== -1) {
                const v = m.knee_travel;
                if (typeof v === "number") {
                    penalty = 14 + Math.min(26, Math.abs(v - t.travel) * 110);
                }
            }
            score -= penalty;
        });

        return Math.max(20, Math.round(score));
    }

    /* ------------------------------------------------------------ graph -- */

    const G = { w: 320, h: 96, pad: 7 };

    function yFor(q) {
        return G.pad + (1 - Math.max(0, Math.min(100, q)) / 100) * (G.h - G.pad * 2);
    }

    if (el.graphBand && el.graphThreshold) {
        const y = yFor(QUALITY_PASS);
        el.graphBand.setAttribute("height", y.toFixed(1));
        el.graphThreshold.setAttribute("y1", y.toFixed(1));
        el.graphThreshold.setAttribute("y2", y.toFixed(1));
    }

    // Catmull-Rom through the points, converted to cubic beziers, so the line
    // reads as a wave rather than a zigzag.
    function smoothPath(pts) {
        if (pts.length === 0) return "";
        if (pts.length === 1) return "M" + pts[0].x + " " + pts[0].y;

        let d = "M" + pts[0].x.toFixed(1) + " " + pts[0].y.toFixed(1);
        for (let i = 0; i < pts.length - 1; i++) {
            const p0 = pts[i - 1] || pts[i];
            const p1 = pts[i];
            const p2 = pts[i + 1];
            const p3 = pts[i + 2] || p2;
            const c1x = p1.x + (p2.x - p0.x) / 6;
            const c1y = p1.y + (p2.y - p0.y) / 6;
            const c2x = p2.x - (p3.x - p1.x) / 6;
            const c2y = p2.y - (p3.y - p1.y) / 6;
            d += "C" + c1x.toFixed(1) + " " + c1y.toFixed(1) +
                "," + c2x.toFixed(1) + " " + c2y.toFixed(1) +
                "," + p2.x.toFixed(1) + " " + p2.y.toFixed(1);
        }
        return d;
    }

    function drawGraph() {
        const data = state.quality.slice(-MAX_POINTS);
        if (!el.graphLine) return;

        if (data.length === 0) {
            el.graphLine.setAttribute("d", "");
            el.graphArea.setAttribute("d", "");
            el.graphDots.innerHTML = "";
            if (el.graphCaption) el.graphCaption.textContent = "Waiting for the first rep";
            if (el.qualityLatest) el.qualityLatest.innerHTML = "&mdash;";
            return;
        }

        const step = data.length === 1 ? 0 : G.w / (data.length - 1);
        const pts = data.map((p, i) => ({
            x: data.length === 1 ? G.w / 2 : i * step,
            y: yFor(p.quality),
            fault: p.fault
        }));

        const line = smoothPath(pts);
        el.graphLine.setAttribute("d", line);
        el.graphArea.setAttribute(
            "d",
            line + "L" + pts[pts.length - 1].x.toFixed(1) + " " + G.h +
            "L" + pts[0].x.toFixed(1) + " " + G.h + "Z"
        );

        el.graphDots.innerHTML = "";
        pts.forEach((p, i) => {
            const isLast = i === pts.length - 1;
            if (!p.fault && !isLast) return;
            const size = isLast ? 7 : 5.4;
            const c = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            c.setAttribute("x", (p.x - size / 2).toFixed(1));
            c.setAttribute("y", (p.y - size / 2).toFixed(1));
            c.setAttribute("width", size);
            c.setAttribute("height", size);
            c.setAttribute("class", "graph-dot" + (p.fault ? " is-fault" : ""));
            el.graphDots.appendChild(c);
        });

        const latest = data[data.length - 1].quality;
        if (el.qualityLatest) el.qualityLatest.textContent = latest + " / 100";

        if (el.graphCaption) {
            const avg = Math.round(
                data.reduce((s, p) => s + p.quality, 0) / data.length
            );
            el.graphCaption.textContent = data.length < MAX_POINTS
                ? "Session average " + avg
                : "Last " + MAX_POINTS + " reps, average " + avg;
        }
    }

    /* ------------------------------------------------------------- feed -- */

    function metricChip(key, value, breached) {
        const spec = METRIC_SPECS[key] || { label: humanise(key), unit: "", digits: 1 };
        const shown = typeof value === "number" ? value.toFixed(spec.digits) : String(value);
        const li = document.createElement("span");
        li.className = "metric" + (breached ? " is-breach" : "");
        li.innerHTML = spec.label + " <b>" + shown + spec.unit + "</b>";
        return li;
    }

    function breachedKeys(faults) {
        const keys = new Set();
        (faults || []).forEach((raw) => {
            const k = String(
                typeof raw === "object" ? (raw.name || raw.fault || "") : raw
            ).toLowerCase();
            if (k.indexOf("depth") !== -1) {
                ["min_knee_angle", "knee_angle", "depth"].forEach((x) => keys.add(x));
            }
            if (k.indexOf("trunk") !== -1) {
                ["max_trunk_angle", "trunk_angle", "trunk_lean"].forEach((x) => keys.add(x));
            }
            if (k.indexOf("travel") !== -1 || k.indexOf("knee") !== -1) {
                keys.add("knee_travel");
            }
        });
        return keys;
    }

    function buildItem(rep) {
        const li = document.createElement("li");
        li.className = "feed-item is-new";
        li.setAttribute("data-verdict", rep.verdict);

        li.dataset.rep = rep.rep;
        if (!rep.text) li.dataset.pending = "1";

        const idx = document.createElement("span");
        idx.className = "feed-index";
        idx.textContent = rep.rep;
        li.appendChild(idx);

        const head = document.createElement("div");
        head.className = "feed-head";

        const verdict = document.createElement("span");
        verdict.className = "feed-verdict";
        verdict.textContent = rep.verdict === "good"
            ? "Clean rep"
            : rep.faults.map(faultLabel).join(" / ");
        head.appendChild(verdict);

        if (typeof rep.latencyMs === "number") {
            const lat = document.createElement("span");
            lat.className = "feed-latency";
            lat.textContent = (rep.latencyMs / 1000).toFixed(1) + "s";
            head.appendChild(lat);
        }
        li.appendChild(head);

        if (rep.text) {
            const p = document.createElement("p");
            p.className = "feed-text";
            p.textContent = rep.text;
            li.appendChild(p);
        }

        const metrics = rep.metrics || {};
        const keys = Object.keys(metrics).filter((k) => metrics[k] !== null && metrics[k] !== undefined);
        if (keys.length) {
            const wrap = document.createElement("div");
            wrap.className = "feed-metrics";
            const breached = breachedKeys(rep.faults);
            keys.slice(0, 4).forEach((k) => {
                wrap.appendChild(metricChip(k, metrics[k], breached.has(k)));
            });
            li.appendChild(wrap);
        }

        return li;
    }

    /* -------------------------------------------------------------- api -- */

    const CoachUI = {

        setStatus(stateName, label) {
            if (!el.conn) return;
            el.conn.setAttribute("data-state", stateName);
            const text = label || {
                live: "Connected",
                connecting: "Connecting",
                offline: "Backend offline"
            }[stateName] || stateName;
            el.connLabel.textContent = text;
        },

        hideHint() {
            if (el.hint) el.hint.hidden = true;
        },

        setHint(text) {
            if (!el.hint) return;
            el.hint.hidden = false;
            el.hint.textContent = text;
        },

        setExercise(name) {
            if (el.sessionExercise) el.sessionExercise.textContent = name;
        },

        setReps(n) {
            state.reps = n;
            if (!el.count) return;
            el.count.textContent = n;
            if (el.countWord) el.countWord.textContent = n === 1 ? "rep" : "reps";
            if (el.countBox) {
                el.countBox.classList.remove("ticked");
                void el.countBox.offsetWidth;
                el.countBox.classList.add("ticked");
            }
        },

        setCoaching(text, verdict) {
            if (el.coaching) el.coaching.textContent = text;
            if (verdict && el.lastBox) {
                el.lastBox.setAttribute("data-verdict", verdict);
                el.lastVerdict.textContent = verdict === "good" ? "Clean rep" : "Adjust";
            }
        },

        pushRep(payload) {
            const faults = payload.faults || [];
            const verdict = faults.length === 0 ? "good" : "fault";
            const exercise = payload.exercise ||
                (el.exerciseSelect ? el.exerciseSelect.value : "squat");

            const rep = {
                rep: payload.rep != null ? payload.rep : state.reps + 1,
                verdict: verdict,
                faults: faults,
                text: payload.text || "",
                metrics: payload.metrics || {},
                latencyMs: payload.latencyMs,
                quality: typeof payload.quality === "number"
                    ? payload.quality
                    : estimateQuality(faults, payload.metrics, exercise)
            };

            if (!state.startedAt) state.startedAt = Date.now();

            this.setReps(rep.rep);
            this.setExercise(exercise);
            const placeholder = verdict === "good"
                ? "Clean rep."
                : rep.faults.map(faultLabel).join(" / ");
            this.setCoaching(rep.text || placeholder, verdict);

            if (verdict === "good") state.good += 1; else state.fault += 1;
            if (el.tallyGood) el.tallyGood.textContent = state.good;
            if (el.tallyFault) el.tallyFault.textContent = state.fault;

            state.quality.push({ quality: rep.quality, fault: verdict === "fault" });
            drawGraph();

            if (el.feed) {
                if (el.feedEmpty) el.feedEmpty.hidden = true;
                const item = buildItem(rep);
                el.feed.insertBefore(item, el.feed.firstChild);
                setTimeout(() => item.classList.remove("is-new"), 600);
                while (el.feed.children.length > 60) {
                    el.feed.removeChild(el.feed.lastChild);
                }
            }

            this.hideHint();
            return rep;
        },

        applyCue(rep, text, latencyMs) {
            if (text) this.setCoaching(text);
            if (!el.feed) return;

            const item = el.feed.querySelector('[data-rep="' + rep + '"]');
            if (!item) return;

            if (text) {
                let p = item.querySelector(".feed-text");
                if (!p) {
                    p = document.createElement("p");
                    p.className = "feed-text";
                    item.insertBefore(p, item.querySelector(".feed-metrics"));
                }
                p.textContent = text;
            }

            if (typeof latencyMs === "number") {
                let lat = item.querySelector(".feed-latency");
                if (!lat) {
                    lat = document.createElement("span");
                    lat.className = "feed-latency";
                    const head = item.querySelector(".feed-head");
                    if (head) head.appendChild(lat);
                }
                lat.textContent = (latencyMs / 1000).toFixed(1) + "s";
            }

            item.removeAttribute("data-pending");
        },

        reset() {
            state.reps = 0;
            state.good = 0;
            state.fault = 0;
            state.quality = [];
            state.startedAt = null;
            this.setReps(0);
            if (el.tallyGood) el.tallyGood.textContent = "0";
            if (el.tallyFault) el.tallyFault.textContent = "0";
            if (el.sessionTime) el.sessionTime.textContent = "0:00";
            if (el.feed) el.feed.innerHTML = "";
            if (el.feedEmpty) el.feedEmpty.hidden = false;
            if (el.lastBox) el.lastBox.setAttribute("data-verdict", "idle");
            if (el.lastVerdict) el.lastVerdict.textContent = "Ready";
            drawGraph();
        }
    };

    if (el.clearFeed) {
        el.clearFeed.addEventListener("click", () => CoachUI.reset());
    }

    if (el.exerciseSelect) {
        el.exerciseSelect.addEventListener("change", (e) => {
            CoachUI.setExercise(e.target.selectedOptions[0].textContent.toLowerCase());
        });
        CoachUI.setExercise(el.exerciseSelect.selectedOptions[0].textContent.toLowerCase());
    }

    drawGraph();
    window.CoachUI = CoachUI;
})();