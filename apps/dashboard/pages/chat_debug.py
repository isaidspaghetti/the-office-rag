from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.fusion import rrf_fuse
from rag.query_expansion import QueryExpansionConfig, build_expander_llm, expand_queries

from apps.dashboard.shared import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LOCAL_DERIVED_PERSIST_DIRS,
    DEFAULT_LOCAL_SCRIPT_PERSIST_DIRS,
    REPO_ROOT,
    _HAS_QDRANT,
    _answer_prompt,
    _build_chroma,
    _build_qdrant,
    _env_or_secret,
    _format_doc_line,
    _mmr_pairs,
    _openai_ready,
    _resolve_under_repo,
    _short_path,
    _similarity_pairs,
)


def render_chat_debug() -> None:
    st.header("Chat & Debug")
    load_dotenv()

    openai_ok, openai_err = _openai_ready()
    qdrant_url = _env_or_secret("QDRANT_URL")
    qdrant_api_key = _env_or_secret("QDRANT_API_KEY")
    qdrant_scripts_collection = _env_or_secret("QDRANT_SCRIPTS_COLLECTION", "office_scripts")
    qdrant_derived_collection = _env_or_secret("QDRANT_DERIVED_COLLECTION", "office_derived_cards")

    default_backend = "qdrant" if qdrant_url else "chroma"

    # --- Query expansion state sync (sidebar <-> quick rerun panel) ---
    # Streamlit reruns on every widget interaction. We keep the left sidebar controls and
    # the "Re-run last question" controls in sync via session_state.
    _QE_STATE_MAP: List[Tuple[str, str]] = [
        ("chat_sidebar_qe_enabled", "chat_quick_qe_enabled"),
        ("chat_sidebar_expand_n", "chat_quick_expand_n"),
        ("chat_sidebar_expand_model", "chat_quick_expand_model"),
        ("chat_sidebar_k_per_query", "chat_quick_k_per_query"),
        ("chat_sidebar_rrf_k0", "chat_quick_rrf_k0"),
        ("chat_sidebar_expand_cache", "chat_quick_expand_cache"),
    ]

    def _qe_mark_sidebar_dirty() -> None:
        st.session_state.chat_qe_dirty_source = "sidebar"

    def _qe_mark_quick_dirty() -> None:
        st.session_state.chat_qe_dirty_source = "quick"
        # Keep the quick panel open while tweaking.
        st.session_state.chat_quick_expanded = True

    # Seed defaults (only if missing).
    st.session_state.setdefault("chat_qe_dirty_source", None)
    st.session_state.setdefault("chat_sidebar_qe_enabled", False)
    st.session_state.setdefault("chat_sidebar_expand_n", 5)
    st.session_state.setdefault("chat_sidebar_expand_model", "gpt-4.1-nano")
    st.session_state.setdefault("chat_sidebar_k_per_query", 6)
    st.session_state.setdefault("chat_sidebar_rrf_k0", 60)
    st.session_state.setdefault(
        "chat_sidebar_expand_cache", "experiments/cache/query_expansion_cache.json"
    )
    st.session_state.setdefault("chat_quick_expanded", False)

    # Default quick values to match sidebar.
    for sb_key, quick_key in _QE_STATE_MAP:
        if quick_key not in st.session_state:
            st.session_state[quick_key] = st.session_state.get(sb_key)

    # One-way sync based on which side changed last.
    src = st.session_state.get("chat_qe_dirty_source")
    if src == "quick":
        for sb_key, quick_key in _QE_STATE_MAP:
            if quick_key in st.session_state:
                st.session_state[sb_key] = st.session_state.get(quick_key)
        st.session_state.chat_qe_dirty_source = None
    elif src == "sidebar":
        for sb_key, quick_key in _QE_STATE_MAP:
            if sb_key in st.session_state:
                st.session_state[quick_key] = st.session_state.get(sb_key)
        st.session_state.chat_qe_dirty_source = None
    backend = st.sidebar.selectbox(
        "Retrieval backend",
        options=["auto", "chroma", "qdrant"],
        index=0,
        help="Auto selects Qdrant if QDRANT_URL is set; otherwise uses local Chroma.",
    )
    if backend == "auto":
        backend = default_backend

    st.sidebar.subheader("Retrieval")
    retrieval_policy = st.sidebar.selectbox(
        "Policy",
        options=["script_only", "derived_only", "derived_then_script"],
        index=2,
    )
    search_type = st.sidebar.selectbox(
        "Search type",
        options=["similarity", "mmr"],
        index=0,
        help="MMR trades off relevance vs diversity. Use lambda closer to 1.0 for more relevance.",
    )
    k = int(st.sidebar.slider("Top-k", min_value=1, max_value=20, value=12))
    fetch_k: Optional[int] = None
    lambda_mult = 0.7
    if search_type == "mmr":
        fetch_k = int(
            st.sidebar.slider(
                "MMR fetch_k",
                min_value=max(8, int(k)),
                max_value=240,
                value=max(24, int(k) * 4),
                help="Candidate pool size MMR selects from (higher = slower, sometimes better).",
            )
        )
        lambda_mult = float(
            st.sidebar.slider(
                "MMR lambda",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="0.0=max diversity, 1.0=max relevance.",
            )
        )
    derived_k = int(st.sidebar.slider("Derived k (routing)", min_value=1, max_value=30, value=12))
    episode_shortlist_size = int(
        st.sidebar.slider("Episode shortlist size", min_value=1, max_value=12, value=6)
    )

    st.sidebar.subheader("Query expansion")
    qe_enabled = bool(
        st.sidebar.checkbox(
            "Expand queries + RRF fuse",
            key="chat_sidebar_qe_enabled",
            on_change=_qe_mark_sidebar_dirty,
            help="Generates alternate retrieval queries and fuses results via Reciprocal Rank Fusion.",
        )
    )
    expand_n = int(
        st.sidebar.slider(
            "Expand n",
            min_value=1,
            max_value=10,
            disabled=(not qe_enabled),
            key="chat_sidebar_expand_n",
            on_change=_qe_mark_sidebar_dirty,
        )
    )
    expand_model = st.sidebar.selectbox(
        "Expand model",
        options=["gpt-4.1-nano", "gpt-4.1-mini", "gpt-5-nano", "gpt-5-mini"],
        disabled=(not qe_enabled),
        key="chat_sidebar_expand_model",
        on_change=_qe_mark_sidebar_dirty,
    )
    k_per_query = int(
        st.sidebar.slider(
            "k per query",
            min_value=1,
            max_value=20,
            disabled=(not qe_enabled),
            help="Docs to retrieve per expanded query before fusion.",
            key="chat_sidebar_k_per_query",
            on_change=_qe_mark_sidebar_dirty,
        )
    )
    rrf_k0 = int(
        st.sidebar.slider(
            "RRF k0",
            min_value=1,
            max_value=200,
            disabled=(not qe_enabled),
            help="Higher reduces the impact of rank differences.",
            key="chat_sidebar_rrf_k0",
            on_change=_qe_mark_sidebar_dirty,
        )
    )
    expand_cache_path = st.sidebar.text_input(
        "Expansion cache",
        disabled=(not qe_enabled),
        help="Set empty to disable caching.",
        key="chat_sidebar_expand_cache",
        on_change=_qe_mark_sidebar_dirty,
    )

    st.sidebar.subheader("Models")
    embed_model = DEFAULT_EMBED_MODEL
    llm_enabled = bool(st.sidebar.checkbox("Generate answer (LLM)", value=True))
    llm_model = st.sidebar.selectbox(
        "LLM model",
        options=[DEFAULT_LLM_MODEL, "gpt-4.1-nano", "gpt-5-mini", "gpt-5-nano"],
        index=0,
    )

    temperature = float(st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0))

    # Both answer-generation and query-expansion require OpenAI.
    if (llm_enabled or qe_enabled) and not openai_ok:
        st.warning(openai_err)
        llm_enabled = False
        qe_enabled = False

    # Local Chroma config
    script_persist_default = DEFAULT_LOCAL_SCRIPT_PERSIST_DIRS[0]
    derived_persist_default = DEFAULT_LOCAL_DERIVED_PERSIST_DIRS[0]
    all_local_script = [
        p for p in DEFAULT_LOCAL_SCRIPT_PERSIST_DIRS if (_resolve_under_repo(p)).exists()
    ]
    all_local_derived = [
        p for p in DEFAULT_LOCAL_DERIVED_PERSIST_DIRS if (_resolve_under_repo(p)).exists()
    ]
    # Auto-discover other chroma_db* dirs under db/
    try:
        for p in sorted((REPO_ROOT / "db").glob("chroma_db*")):
            rel = _short_path(p)
            if rel.startswith("db/") and rel not in all_local_script:
                all_local_script.append(rel)
            if "derived" in rel and rel not in all_local_derived:
                all_local_derived.append(rel)
    except Exception:
        pass

    if backend == "chroma":
        st.caption("Backend: local Chroma (persisted under db/)")
        script_persist = st.sidebar.selectbox(
            "Script Index",
            options=(all_local_script or DEFAULT_LOCAL_SCRIPT_PERSIST_DIRS),
            index=0,
        )
        derived_persist = st.sidebar.selectbox(
            "Derived Data Index",
            options=(all_local_derived or DEFAULT_LOCAL_DERIVED_PERSIST_DIRS),
            index=0,
        )
    else:
        st.caption("Backend: Qdrant (cloud)")
        if not _HAS_QDRANT:
            st.error("Qdrant backend is not available (missing dependencies).")
            return
        if not qdrant_url:
            st.error("Missing QDRANT_URL (set in Streamlit secrets or env).")
            return
        st.sidebar.text_input("Qdrant URL", value=qdrant_url, disabled=True)
        st.sidebar.text_input(
            "Scripts collection", value=str(qdrant_scripts_collection), disabled=True
        )
        st.sidebar.text_input(
            "Derived collection", value=str(qdrant_derived_collection), disabled=True
        )
        script_persist = script_persist_default
        derived_persist = derived_persist_default

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_last_question" not in st.session_state:
        st.session_state.chat_last_question = ""

    # Render history
    for m in st.session_state.chat_messages:
        role = str(m.get("role") or "assistant")
        content = str(m.get("content") or "")
        with st.chat_message(role):
            st.markdown(content)

    question_to_run: Optional[str] = None

    new_question = st.chat_input("Ask about The Office…")
    if new_question:
        question_to_run = str(new_question)
        st.session_state.chat_last_question = str(new_question)
        st.session_state.chat_messages.append({"role": "user", "content": question_to_run})
        with st.chat_message("user"):
            st.markdown(question_to_run)

    # Inline quick rerun controls (main pane) for the last asked question.
    last_question = str(st.session_state.get("chat_last_question") or "").strip()
    quick_backend = backend
    quick_retrieval_policy = retrieval_policy
    quick_search_type = str(search_type)
    quick_k = int(k)
    quick_fetch_k = int(fetch_k) if isinstance(fetch_k, int) else max(24, int(k) * 4)
    quick_lambda_mult = float(lambda_mult)
    quick_derived_k = int(derived_k)
    quick_episode_shortlist_size = int(episode_shortlist_size)
    quick_qe_enabled = bool(qe_enabled)
    quick_expand_n = int(expand_n)
    quick_expand_model = str(expand_model)
    quick_k_per_query = int(k_per_query)
    quick_rrf_k0 = int(rrf_k0)
    quick_expand_cache_path = str(expand_cache_path)
    quick_llm_enabled = bool(llm_enabled)
    quick_llm_model = str(llm_model)
    quick_temperature = float(temperature)

    if last_question:
        # Streamlit reruns on every widget interaction. Persist expander-open state so it
        # doesn't collapse while you're actively tweaking settings.
        if "chat_quick_expanded" not in st.session_state:
            st.session_state.chat_quick_expanded = False
        if "chat_quick_snapshot" not in st.session_state:
            st.session_state.chat_quick_snapshot = {}

        _quick_keys = [
            "chat_quick_question",
            "chat_quick_backend",
            "chat_quick_policy",
            "chat_quick_search_type",
            "chat_quick_k",
            "chat_quick_derived_k",
            "chat_quick_shortlist",
            "chat_quick_fetch_k",
            "chat_quick_lambda",
            "chat_quick_qe_enabled",
            "chat_quick_expand_n",
            "chat_quick_k_per_query",
            "chat_quick_expand_model",
            "chat_quick_rrf_k0",
            "chat_quick_expand_cache",
            "chat_quick_llm_enabled",
            "chat_quick_llm_model",
            "chat_quick_temp",
        ]
        prev_snapshot = st.session_state.get("chat_quick_snapshot") or {}
        cur_snapshot_pre = {
            k: st.session_state.get(k) for k in _quick_keys if k in st.session_state
        }
        if prev_snapshot and cur_snapshot_pre and cur_snapshot_pre != prev_snapshot:
            st.session_state.chat_quick_expanded = True

        with st.expander(
            "Re-run last question with different params",
            expanded=bool(st.session_state.chat_quick_expanded),
        ):
            st.caption(
                "This lets you tweak settings right in chat without touching the left sidebar."
            )
            quick_question = st.text_input(
                "Question", value=last_question, key="chat_quick_question"
            )
            qc1, qc2, qc3 = st.columns(3)
            with qc1:
                quick_backend = st.selectbox(
                    "Backend",
                    options=["chroma", "qdrant"],
                    index=(0 if backend == "chroma" else 1),
                    key="chat_quick_backend",
                )
                quick_retrieval_policy = st.selectbox(
                    "Routing Policy",
                    options=["script_only", "derived_only", "derived_then_script"],
                    index=["script_only", "derived_only", "derived_then_script"].index(
                        retrieval_policy
                    ),
                    key="chat_quick_policy",
                )
                quick_search_type = st.selectbox(
                    "Search type",
                    options=["similarity", "mmr"],
                    index=["similarity", "mmr"].index(str(search_type)),
                    key="chat_quick_search_type",
                )
            with qc2:
                quick_k = int(st.slider("Top-k", 1, 20, int(k), key="chat_quick_k"))
                quick_derived_k = int(
                    st.slider("Derived k", 1, 30, int(derived_k), key="chat_quick_derived_k")
                )
            with qc3:
                quick_episode_shortlist_size = int(
                    st.slider(
                        "Shortlist size",
                        1,
                        12,
                        int(episode_shortlist_size),
                        key="chat_quick_shortlist",
                    )
                )
                quick_llm_enabled = bool(
                    st.checkbox(
                        "Generate answer (LLM)",
                        value=bool(llm_enabled),
                        key="chat_quick_llm_enabled",
                    )
                )

            qret1, qret2 = st.columns(2)
            with qret1:
                quick_fetch_k = int(
                    st.slider(
                        "MMR fetch_k",
                        min_value=max(8, int(quick_k)),
                        max_value=240,
                        value=int(quick_fetch_k),
                        disabled=(str(quick_search_type) != "mmr"),
                        key="chat_quick_fetch_k",
                    )
                )
            with qret2:
                quick_lambda_mult = float(
                    st.slider(
                        "MMR lambda",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(quick_lambda_mult),
                        disabled=(str(quick_search_type) != "mmr"),
                        key="chat_quick_lambda",
                    )
                )

            st.markdown("---")
            st.caption("Optional: query expansion + fusion")
            qxe1, qxe2 = st.columns(2)
            with qxe1:
                quick_qe_enabled = bool(
                    st.checkbox(
                        "Expand queries + RRF fuse",
                        key="chat_quick_qe_enabled",
                        on_change=_qe_mark_quick_dirty,
                    )
                )
                quick_expand_n = int(
                    st.slider(
                        "Expand n",
                        min_value=1,
                        max_value=10,
                        disabled=(not bool(quick_qe_enabled)),
                        key="chat_quick_expand_n",
                        on_change=_qe_mark_quick_dirty,
                    )
                )
                quick_k_per_query = int(
                    st.slider(
                        "k per query",
                        min_value=1,
                        max_value=20,
                        disabled=(not bool(quick_qe_enabled)),
                        key="chat_quick_k_per_query",
                        on_change=_qe_mark_quick_dirty,
                    )
                )
            with qxe2:
                quick_expand_model = st.text_input(
                    "Expand model",
                    disabled=(not bool(quick_qe_enabled)),
                    key="chat_quick_expand_model",
                    on_change=_qe_mark_quick_dirty,
                )
                quick_rrf_k0 = int(
                    st.slider(
                        "RRF k0",
                        min_value=1,
                        max_value=200,
                        disabled=(not bool(quick_qe_enabled)),
                        key="chat_quick_rrf_k0",
                        on_change=_qe_mark_quick_dirty,
                    )
                )
                quick_expand_cache_path = st.text_input(
                    "Expansion cache",
                    disabled=(not bool(quick_qe_enabled)),
                    key="chat_quick_expand_cache",
                    on_change=_qe_mark_quick_dirty,
                )

            qcm1, qcm2 = st.columns([0.7, 0.3])
            with qcm1:
                quick_llm_model = st.text_input(
                    "LLM model", value=str(llm_model), key="chat_quick_llm_model"
                )
            with qcm2:
                quick_temperature = float(
                    st.slider(
                        "Temp",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(temperature),
                        key="chat_quick_temp",
                    )
                )

            run_quick = st.button("Re-run in chat", key="chat_quick_rerun")
            if run_quick:
                question_to_run = str(quick_question).strip()
                if question_to_run:
                    st.session_state.chat_last_question = question_to_run
                    st.session_state.chat_messages.append(
                        {"role": "user", "content": question_to_run}
                    )
                    with st.chat_message("user"):
                        st.markdown(question_to_run)

            cc1, cc2 = st.columns([0.7, 0.3])
            with cc1:
                st.caption("Tip: changing values reruns the app; this panel should stay open now.")
            with cc2:
                if st.button("Collapse", key="chat_quick_collapse"):
                    st.session_state.chat_quick_expanded = False
                    st.rerun()

            # Update snapshot after widgets are created so future changes keep the panel open.
            st.session_state.chat_quick_snapshot = {
                k: st.session_state.get(k) for k in _quick_keys if k in st.session_state
            }

    if not question_to_run:
        with st.expander("Setup / expectations"):
            st.write(
                "Local: uses Chroma under `db/`. Deployed: auto-switches to Qdrant when QDRANT_URL is set."
            )
            st.write("Policy `derived_then_script` uses derived cards to route into script chunks.")
        return

    # If quick rerun was used, override sidebar settings for this execution only.
    if (
        str(st.session_state.get("chat_last_question") or "").strip()
        == str(question_to_run).strip()
    ):
        if "chat_quick_backend" in st.session_state:
            backend = str(quick_backend)
        if "chat_quick_policy" in st.session_state:
            retrieval_policy = str(quick_retrieval_policy)
        if "chat_quick_search_type" in st.session_state:
            search_type = str(quick_search_type)
        if "chat_quick_k" in st.session_state:
            k = int(quick_k)
        if "chat_quick_fetch_k" in st.session_state:
            fetch_k = int(quick_fetch_k)
        if "chat_quick_lambda" in st.session_state:
            lambda_mult = float(quick_lambda_mult)
        if "chat_quick_derived_k" in st.session_state:
            derived_k = int(quick_derived_k)
        if "chat_quick_shortlist" in st.session_state:
            episode_shortlist_size = int(quick_episode_shortlist_size)
        if "chat_quick_qe_enabled" in st.session_state:
            qe_enabled = bool(quick_qe_enabled)
        if "chat_quick_expand_n" in st.session_state:
            expand_n = int(quick_expand_n)
        if "chat_quick_expand_model" in st.session_state:
            expand_model = str(quick_expand_model)
        if "chat_quick_k_per_query" in st.session_state:
            k_per_query = int(quick_k_per_query)
        if "chat_quick_rrf_k0" in st.session_state:
            rrf_k0 = int(quick_rrf_k0)
        if "chat_quick_expand_cache" in st.session_state:
            expand_cache_path = str(quick_expand_cache_path)
        if "chat_quick_llm_enabled" in st.session_state:
            llm_enabled = bool(quick_llm_enabled)
        if "chat_quick_llm_model" in st.session_state:
            llm_model = str(quick_llm_model)
        if "chat_quick_temp" in st.session_state:
            temperature = float(quick_temperature)

    # Guardrail: query expansion needs OpenAI even if LLM answering is off.
    if qe_enabled and not openai_ok:
        st.warning(openai_err)
        qe_enabled = False

    # Build vectorstores lazily per request (simple + robust; can be cached later)
    try:
        with st.spinner("Initializing vectorstores…"):
            if backend == "chroma":
                script_db = _build_chroma(
                    _resolve_under_repo(script_persist), embed_model=embed_model
                )
                derived_db = _build_chroma(
                    _resolve_under_repo(derived_persist), embed_model=embed_model
                )
            else:
                script_db = _build_qdrant(
                    url=str(qdrant_url),
                    api_key=qdrant_api_key,
                    collection=str(qdrant_scripts_collection),
                    embed_model=embed_model,
                )
                derived_db = _build_qdrant(
                    url=str(qdrant_url),
                    api_key=qdrant_api_key,
                    collection=str(qdrant_derived_collection),
                    embed_model=embed_model,
                )
    except Exception as e:
        st.error(f"Failed to init vectorstore: {type(e).__name__}: {e}")
        return

    # Retrieval
    t0 = time.time()
    retrieved: List[Tuple[Any, Optional[float]]] = []
    routing: Optional[Dict[str, Any]] = None
    expanded_queries: Optional[List[str]] = None
    qe_details: Optional[Dict[str, Any]] = None
    qe_error: Optional[str] = None

    def _retrieve(db: Any, question: str, *, k_docs: int) -> List[Tuple[Any, Optional[float]]]:
        if str(search_type) == "mmr":
            return _mmr_pairs(
                db,
                question,
                k=int(k_docs),
                fetch_k=(int(fetch_k) if isinstance(fetch_k, int) else None),
                lambda_mult=float(lambda_mult),
            )
        return _similarity_pairs(db, question, k=int(k_docs))

    try:
        with st.spinner("Retrieving context…"):
            # Optionally expand the question into multiple retrieval queries.
            if qe_enabled:
                try:
                    cache_path = (
                        Path(str(expand_cache_path))
                        if str(expand_cache_path or "").strip()
                        else None
                    )
                    expand_cfg = QueryExpansionConfig(
                        enabled=True,
                        n=int(expand_n),
                        model=str(expand_model),
                        temperature=0.0,
                        cache_path=cache_path,
                    )
                    expander_llm = build_expander_llm(expand_cfg)
                    expanded_queries = expand_queries(
                        question=question_to_run, llm=expander_llm, config=expand_cfg
                    )
                except Exception as e:
                    # Fall back to base question if expansion fails.
                    expanded_queries = [str(question_to_run)]
                    qe_error = f"{type(e).__name__}: {e}"
            else:
                expanded_queries = [str(question_to_run)]

            per_q_k = int(k_per_query) if qe_enabled else int(k)
            per_q_k = max(1, min(50, per_q_k))

            def _fuse_or_single(db: Any, *, question: str) -> List[Tuple[Any, Optional[float]]]:
                """Return top-k results, optionally using expanded queries + RRF."""
                if not qe_enabled or not expanded_queries or len(expanded_queries) <= 1:
                    return _retrieve(db, question, k_docs=int(k))

                per_query: Dict[str, List[Tuple[Any, Optional[float]]]] = {}
                for q in expanded_queries:
                    per_query[q] = _retrieve(db, q, k_docs=int(per_q_k))
                fused = rrf_fuse(per_query, k0=int(rrf_k0))
                return [(fd.doc, fd.fused_score) for fd in fused[: int(k)]]

            if retrieval_policy == "script_only":
                retrieved = _fuse_or_single(script_db, question=question_to_run)
            elif retrieval_policy == "derived_only":
                retrieved = _fuse_or_single(derived_db, question=question_to_run)
            else:
                # derived -> script routing (keep routing stage similarity-based for stability)
                derived_pairs = _similarity_pairs(derived_db, question_to_run, k=int(derived_k))
                shortlist: List[str] = []
                seen: set[str] = set()
                for d, _s in derived_pairs:
                    meta = getattr(d, "metadata", None) or {}
                    eid = str(meta.get("episode_id") or "").strip().upper()
                    if not eid:
                        # derived episode cards also have an identity line like "Episode card: S02E12 — ..."
                        txt = getattr(d, "page_content", None) or ""
                        for tok in str(txt).split():
                            if len(tok) == 6 and tok.upper().startswith("S") and "E" in tok.upper():
                                eid = tok.strip().upper().strip(",.;:()[]{}")
                                break
                    if not eid or eid in seen:
                        continue
                    seen.add(eid)
                    shortlist.append(eid)
                    if len(shortlist) >= int(episode_shortlist_size):
                        break

                def _retrieve_script_filtered(
                    q: str, *, k_docs: int
                ) -> List[Tuple[Any, Optional[float]]]:
                    if not shortlist:
                        return _retrieve(script_db, q, k_docs=int(k_docs))

                    allowed = {s.strip().upper() for s in shortlist}
                    candidate_k = max(int(k_docs) * 12, 60)
                    if str(search_type) == "mmr":
                        # Try grabbing a larger MMR set, then filter down.
                        pairs = _mmr_pairs(
                            script_db,
                            q,
                            k=int(max(int(k_docs) * 5, 40)),
                            fetch_k=(int(fetch_k) if isinstance(fetch_k, int) else None),
                            lambda_mult=float(lambda_mult),
                        )
                    else:
                        pairs = _similarity_pairs(script_db, q, k=int(candidate_k))

                    filtered: List[Tuple[Any, Optional[float]]] = []
                    for d, s in pairs:
                        meta = getattr(d, "metadata", None) or {}
                        eid = str(meta.get("episode_id") or "").strip().upper()
                        if eid and eid in allowed:
                            filtered.append((d, s))
                            if len(filtered) >= int(k_docs):
                                break
                    if filtered:
                        return filtered

                    if str(search_type) == "mmr":
                        # Prefer a similarity fall-back when filtered MMR yields nothing.
                        sim_pairs = _similarity_pairs(script_db, q, k=int(candidate_k))
                        filtered2: List[Tuple[Any, Optional[float]]] = []
                        for d, s in sim_pairs:
                            meta = getattr(d, "metadata", None) or {}
                            eid = str(meta.get("episode_id") or "").strip().upper()
                            if eid and eid in allowed:
                                filtered2.append((d, s))
                                if len(filtered2) >= int(k_docs):
                                    break
                        if filtered2:
                            return filtered2

                    # Last resort: return unfiltered results.
                    return _retrieve(script_db, q, k_docs=int(k_docs))

                if not qe_enabled or not expanded_queries or len(expanded_queries) <= 1:
                    retrieved = _retrieve_script_filtered(question_to_run, k_docs=int(k))
                else:
                    per_query2: Dict[str, List[Tuple[Any, Optional[float]]]] = {}
                    for q in expanded_queries:
                        per_query2[q] = _retrieve_script_filtered(q, k_docs=int(per_q_k))
                    fused2 = rrf_fuse(per_query2, k0=int(rrf_k0))
                    retrieved = [(fd.doc, fd.fused_score) for fd in fused2[: int(k)]]

                routing = {
                    "policy": "derived_then_script",
                    "episode_shortlist": shortlist,
                    "search_type": str(search_type),
                    "mmr": (
                        {
                            "fetch_k": int(fetch_k) if isinstance(fetch_k, int) else None,
                            "lambda": float(lambda_mult),
                        }
                        if str(search_type) == "mmr"
                        else None
                    ),
                    "query_expansion": (
                        {
                            "enabled": True,
                            "n": int(expand_n),
                            "model": str(expand_model),
                            "k_per_query": int(per_q_k),
                            "fusion": "rrf",
                            "rrf_k0": int(rrf_k0),
                        }
                        if qe_enabled
                        else {"enabled": False}
                    ),
                    "derived_top": [
                        {
                            "rank": i + 1,
                            "episode_id": str(
                                (getattr(d, "metadata", None) or {}).get("episode_id") or ""
                            ).strip(),
                            "score": (float(s) if s is not None else None),
                            "source": str(
                                (getattr(d, "metadata", None) or {}).get("source_file")
                                or (getattr(d, "metadata", None) or {}).get("source")
                                or ""
                            ),
                        }
                        for i, (d, s) in enumerate(derived_pairs[: min(len(derived_pairs), 12)])
                    ],
                }

            # Record query-expansion details for display even when not using derived routing.
            if qe_enabled:
                qe_details = {
                    "enabled": True,
                    "n": int(expand_n),
                    "model": str(expand_model),
                    "k_per_query": int(per_q_k),
                    "fusion": "rrf",
                    "rrf_k0": int(rrf_k0),
                    "cache": (
                        str(expand_cache_path) if str(expand_cache_path or "").strip() else None
                    ),
                    "queries": list(expanded_queries or []),
                    "error": qe_error,
                }
    except Exception as e:
        st.error(f"Retrieval failed: {type(e).__name__}: {e}")
        retrieved = []

    retrieval_ms = int((time.time() - t0) * 1000)

    if qe_enabled and expanded_queries:
        with st.expander(f"Expanded queries ({len(expanded_queries)} total)", expanded=False):
            st.caption(
                "These are the retrieval queries generated from your question (first is always the original)."
            )
            if isinstance(qe_details, dict) and qe_details.get("error"):
                st.warning(
                    f"Query expansion fell back to the original question: {qe_details.get('error')}"
                )
            st.text_area(
                "Queries",
                value="\n".join([f"{i+1}. {q}" for i, q in enumerate(expanded_queries)]),
                height=140,
            )
            if isinstance(qe_details, dict):
                cfg = {k: v for k, v in qe_details.items() if k not in {"queries"}}
                if cfg:
                    st.caption("Expansion config")
                    st.json(cfg)

    with st.expander(
        f"Retrieved context ({len(retrieved)} docs, {retrieval_ms}ms)", expanded=False
    ):
        if routing:
            st.write("Routing:")
            st.json(routing)
        for i, (d, s) in enumerate(retrieved, start=1):
            st.markdown(f"**#{i}**  score={s if s is not None else 'n/a'}")
            st.write(_format_doc_line(d))
            st.text((getattr(d, "page_content", None) or "")[:900])

    context = "\n\n---\n\n".join(
        [str(getattr(d, "page_content", None) or "") for (d, _s) in retrieved]
    )

    answer_text = ""
    if not llm_enabled:
        answer_text = "(LLM disabled; retrieval only.)"
    else:
        try:
            with st.spinner("Generating answer…"):
                llm = ChatOpenAI(model=str(llm_model), temperature=float(temperature))
                sys, user = _answer_prompt(question=question_to_run, context=context)
                msg = llm.invoke([("system", sys), ("human", user)])
                answer_text = str(getattr(msg, "content", "") or "").strip()
        except Exception as e:
            answer_text = f"(LLM error: {type(e).__name__}: {e})"

    st.session_state.chat_messages.append({"role": "assistant", "content": answer_text})
    with st.chat_message("assistant"):
        st.markdown(answer_text or "(empty)")

    if st.sidebar.button("Clear chat"):
        st.session_state.chat_messages = []
        st.rerun()
