# main.py (final corrected full file)
import streamlit as st
import os
import re
import datetime
from typing import Optional
from dotenv import load_dotenv

# optional import; if streamlit_mermaid isn't available, ignore it
try:
    from streamlit_mermaid import st_mermaid  # noqa: F401
except Exception:
    st_mermaid = None  # type: ignore

from qa_pipeline import (
    get_advanced_retriever, get_metadata_retriever, build_single_answer_chain,
    build_multi_answer_chain, build_steps_answer_chain, build_flowchart_answer_chain,
    build_general_chat_chain, choose_model
)
from db_handler import get_vectorstore, store_chunks, delete_vectorstore_for_session
from document_processor import (
    extract_text_from_pdf, extract_text_from_docx, split_into_chunks,
    extract_text_from_url
)
from history_store import (
    load_all_chat_sessions, save_chat_session, create_new_chat_session, delete_chat_session
)

# Load environment variables
load_dotenv()
st.set_page_config(page_title="QA Bot", layout="wide", initial_sidebar_state="expanded")


# ---------------- Helpers ---------------- #
def parse_multi_answer_output(llm_output):
    answers = []
    styles = ["Technical", "Summary", "Simple"]
    header_pattern = "|".join([re.escape(f"**{style} Answer**") for style in styles])
    parts = re.split(f"({header_pattern})", llm_output, flags=re.IGNORECASE)
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            try:
                header, content = parts[i], parts[i + 1]
                style = header.replace("**", "").replace("Answer", "").strip()
                content = re.sub(r'^[=\-—_*\s]+', '', content, flags=re.MULTILINE).strip()
                if style and content:
                    answers.append({"style": style, "content": content})
            except IndexError:
                continue
    return answers


def _safe_float(x, default: Optional[float] = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default if default is not None else 0.0


def rank_docs(docs, query=None):
    scored = []
    now = datetime.datetime.now()   # ✅ FIXED here
    qtokens = set(re.findall(r"\w+", (query or "").lower()))

    for d in docs:
        try:
            meta = getattr(d, "metadata", {}) or {}
            content = getattr(d, "page_content", "") or ""

            score = 0.0
            if "score" in meta:
                score = _safe_float(meta.get("score"), 0.0)
            elif "similarity" in meta:
                score = _safe_float(meta.get("similarity"), 0.0)
            elif "cosine_score" in meta:
                score = _safe_float(meta.get("cosine_score"), 0.0)
            elif "distance" in meta:
                distance = _safe_float(meta.get("distance"), None)
                if distance is not None:
                    score = 1.0 / (1.0 + max(0.0, distance))

            created = meta.get("created_at") or meta.get("createdAt") or meta.get("timestamp")
            if created:
                try:
                    created_dt = datetime.datetime.fromisoformat(created)
                    days = (now - created_dt).days
                    recency_boost = max(0.0, 1.0 - (days / 365.0)) * 0.15
                    score += recency_boost
                except Exception:
                    pass

            if qtokens and content:
                content_tokens = set(re.findall(r"\w+", content.lower()))
                overlap = len(qtokens & content_tokens)
                if overlap:
                    overlap_boost = min(0.25, overlap / max(1, len(qtokens)) * 0.25)
                    score += overlap_boost

            chunk_len = meta.get("chunk_length") or len(content)
            try:
                chunk_len = int(chunk_len)
                if chunk_len < 50:
                    score -= 0.05
            except Exception:
                pass

            scored.append((score, d))
        except Exception:
            scored.append((0.0, d))

    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    return [d for _, d in scored_sorted]


def safe_invoke(chain, inputs):
    try:
        result = chain.invoke(inputs)
        answer = str(result).strip()
        if not answer:
            answer = "⚠️ No answer generated (empty response)."
        return answer
    except Exception as e:
        return f"⚠️ Error: {e}"


def _highlight_query_terms(text: str, query: str) -> str:
    if not text:
        return ""
    if not query:
        return text.replace("\n", "<br>")
    tokens = sorted(set(re.findall(r"\w+", query.lower())), key=len, reverse=True)
    safe = text
    for t in tokens:
        if len(t) < 2:
            continue
        pattern = re.compile(rf'(\b{re.escape(t)}\b)', flags=re.IGNORECASE)
        safe = pattern.sub(r'<mark>\1</mark>', safe)
    return safe.replace("\n", "<br>")


# ---------------- Auth ---------------- #
def check_authentication():
    correct_api_key = os.getenv("APP_API_KEY")
    user_api_key = st.sidebar.text_input("Enter your App API Key:", type="password")
    if user_api_key == correct_api_key:
        return True
    elif user_api_key:
        st.sidebar.error("Incorrect API Key")
        return False
    return False

# ---------------- Main App ---------------- #
def run_app():
    if "session_initialized" not in st.session_state:
        st.session_state.all_chat_sessions = load_all_chat_sessions()
        if not st.session_state.all_chat_sessions:
            new_session = create_new_chat_session()
            st.session_state.all_chat_sessions.append(new_session)
        st.session_state.current_chat_session = st.session_state.all_chat_sessions[0]
        st.session_state.session_initialized = True

    if "distrusted_sources" not in st.session_state:
        st.session_state.distrusted_sources = set()

    if "last_retrieved_docs" not in st.session_state:
        st.session_state.last_retrieved_docs = []

    if "open_source" not in st.session_state:
        st.session_state.open_source = None

    # Sidebar
    with st.sidebar:
        st.success("Authenticated Successfully")
        st.button("➕ New Chat", on_click=lambda: new_chat(), use_container_width=True, type="primary")
        st.markdown("### 💬 Your Chats")
        for session in st.session_state.all_chat_sessions:
            col1, col2 = st.columns([5, 1])
            if col1.button(session["name"], key=f"select_{session['id']}", use_container_width=True):
                st.session_state.current_chat_session = session
                st.rerun()
            col2.button("🗑️", key=f"delete_{session['id']}", on_click=lambda sid=session['id']: delete_chat(sid))

        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        st.session_state.search_mode = st.radio("Search Mode", ("⚡ Fast", "🎯 Deep"), horizontal=True)
        st.session_state.k_retrieval = st.slider("Docs to Retrieve (k)", 1, 10, 4, 1)
        st.session_state.answer_mode = st.radio(
            "Answer Mode", ("Docs only", "General only", "Hybrid (Docs + General)"),
            index=["Docs only", "General only", "Hybrid (Docs + General)"].index(st.session_state.get("answer_mode", "Docs only"))
        )

        st.session_state.answer_display_mode = st.radio(
            "Answer Display", ("Single Answer", "Ranked Answers"),
            index=1
        )

        st.markdown("### 🧠 Model Selection")
        st.session_state.doc_model = st.radio("Docs Model", ("gemini-1.5-flash", "gemini-1.5-pro"), index=0)
        st.session_state.general_model = st.radio("General Model", ("gemini-1.5-flash", "gemini-1.5-pro"), index=0)

    st.header("Chat with your Docs or Ask Anything")

    # Upload
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs, Word docs, or text files",
                                      type=["pdf", "docx", "txt"], accept_multiple_files=True)
    url_input = st.text_input("Or enter a webpage URL")

    if uploaded_files or url_input:
        if st.button("Process & Store Documents", type="primary"):
            all_chunks = []
            for file in uploaded_files:
                try:
                    if file.name.endswith(".pdf"):
                        text = extract_text_from_pdf(file)
                    elif file.name.endswith(".docx"):
                        text = extract_text_from_docx(file)
                    elif file.name.endswith(".txt"):
                        raw = file.read()
                        if isinstance(raw, bytes):
                            text = raw.decode("utf-8", errors="ignore")
                        else:
                            text = str(raw)
                    else:
                        text = ""
                except Exception:
                    text = ""
                if text:
                    all_chunks.extend(split_into_chunks(text, 800, 200, file.name))

            if url_input:
                url_text = extract_text_from_url(url_input)
                if url_text:
                    all_chunks.extend(split_into_chunks(url_text, 800, 200, url_input))

            if all_chunks:
                store_chunks(all_chunks, st.session_state.current_chat_session["id"])
                st.success(f"Stored {len(all_chunks)} chunks.")
            else:
                st.warning("No valid text extracted.")

    st.markdown("---")

    # History display (now includes persisted citations if present)
    if st.session_state.current_chat_session.get("messages"):
        for mid, msg in enumerate(st.session_state.current_chat_session["messages"]):
            # display user question
            st.chat_message("user").markdown(msg.get("question", ""))
            # display assistant response with possible rendered_answers metadata
            with st.chat_message("assistant"):
                if msg.get("rendered_answers"):
                    # iterate through saved rendered answers and render interactive citation buttons
                    for rid, r in enumerate(msg["rendered_answers"]):
                        header = f"**Rank {r['rank']} Answer:**" if r.get("rank") else "**General Answer:**"
                        st.markdown(header)
                        st.markdown(r.get("text", ""))
                        view_key = f"hist_view_{st.session_state.current_chat_session['id']}_{mid}_{rid}"
                        if r.get("source_type") == "document":
                            st.markdown(f"📚 Source: `{r.get('source','unknown')}`  — chunk_id: `{r.get('chunk_id')}`")
                            if st.button("🔗 View source", key=view_key):
                                st.session_state.open_source = {
                                    "source_type": "document",
                                    "source": r.get("source"),
                                    "chunk_id": r.get("chunk_id"),
                                    "content": r.get("content"),
                                    "rank": r.get("rank"),
                                    "highlight_query": msg.get("question", "")
                                }
                                st.rerun()
                        else:
                            src_label = r.get("source", "general")
                            if isinstance(src_label, str) and src_label.lower().startswith("http"):
                                st.markdown(f"🌐 Source (general model): [{src_label}]({src_label})")
                            else:
                                st.markdown(f"🤖 Source: General model ({src_label})")
                            if st.button("🔗 View source", key=view_key):
                                st.session_state.open_source = {
                                    "source_type": "general",
                                    "source": src_label,
                                    "chunk_id": None,
                                    "content": None,
                                    "rank": r.get("rank"),
                                    "highlight_query": msg.get("question", "")
                                }
                                st.rerun()
                else:
                    st.markdown(msg.get("answer", ""))

    # Chat input and generation
    if prompt := st.chat_input("Ask anything..."):
        vectorstore = get_vectorstore(st.session_state.current_chat_session["id"])
        has_docs = vectorstore is not None

        # reset open viewer for new question
        st.session_state.open_source = None

        # append new message placeholder
        st.session_state.current_chat_session["messages"].append({"question": prompt, "answer": ""})
        st.chat_message("user").markdown(prompt)

        answer = ""
        docs = []
        rendered_answers = []

        with st.chat_message("assistant"):
            if st.session_state.answer_mode == "General only":
                general_model = choose_model(prompt, st.session_state.general_model)
                chain = build_general_chat_chain(general_model)
                answer = safe_invoke(chain, {"question": prompt})
                rendered_answers = [{
                    "rank": None,
                    "text": answer,
                    "source_type": "general",
                    "source": st.session_state.general_model,
                    "chunk_id": None,
                    "content": None
                }]
                st.markdown(answer)

            elif st.session_state.answer_mode == "Docs only":
                if not has_docs:
                    answer = "⚠️ Please upload a document first."
                    st.warning(answer)
                else:
                    retriever = get_metadata_retriever(vectorstore) if st.session_state.search_mode == "⚡ Fast" else get_advanced_retriever(vectorstore, st.session_state.k_retrieval)
                    try:
                        docs = retriever.invoke(prompt)
                    except AttributeError:
                        docs = retriever.get_relevant_documents(prompt)

                    st.session_state.last_retrieved_docs = docs

                    docs = rank_docs(docs, prompt)
                    docs = [d for d in docs if d.metadata.get("source") not in st.session_state.distrusted_sources]

                    doc_model = choose_model(prompt, st.session_state.doc_model)
                    if "flowchart" in prompt.lower():
                        chain = build_flowchart_answer_chain(doc_model)
                    elif any(w in prompt.lower() for w in ["how to", "steps", "instructions"]):
                        chain = build_steps_answer_chain(doc_model)
                    elif any(w in prompt.lower() for w in ["what is", "who is", "when was", "list"]):
                        chain = build_single_answer_chain(doc_model)
                    else:
                        chain = build_multi_answer_chain(doc_model)

                    if st.session_state.answer_display_mode == "Ranked Answers":
                        for i, doc in enumerate(docs[:st.session_state.k_retrieval], start=1):
                            context = doc.page_content
                            answer_i = safe_invoke(chain, {"context": context, "question": prompt})
                            rendered_answers.append({
                                "rank": i,
                                "text": answer_i,
                                "source_type": "document",
                                "source": doc.metadata.get("source", "unknown"),
                                "chunk_id": doc.metadata.get("chunk_id"),
                                "content": doc.page_content
                            })
                        # display answers now
                        for r in rendered_answers:
                            st.markdown(f"**Rank {r['rank']} Answer:**\n{r['text']}")
                    else:
                        context = "\n\n".join([d.page_content for d in docs])
                        answer = safe_invoke(chain, {"context": context, "question": prompt})
                        rendered_answers = [{
                            "rank": 1,
                            "text": answer,
                            "source_type": "document",
                            "source": ", ".join(sorted({d.metadata.get("source", "unknown") for d in docs})),
                            "chunk_id": None,
                            "content": context
                        }]
                        st.markdown(answer)

            else:  # Hybrid
                general_model = choose_model(prompt, st.session_state.general_model)
                general_chain = build_general_chat_chain(general_model)
                general_answer = safe_invoke(general_chain, {"question": prompt})

                general_rendered = {
                    "rank": None,
                    "text": general_answer,
                    "source_type": "general",
                    "source": st.session_state.general_model,
                    "chunk_id": None,
                    "content": None
                }

                if has_docs:
                    retriever = get_metadata_retriever(vectorstore) if st.session_state.search_mode == "⚡ Fast" else get_advanced_retriever(vectorstore, st.session_state.k_retrieval)
                    try:
                        docs = retriever.get_relevant_documents(prompt)
                    except Exception:
                        docs = retriever.invoke(prompt)

                    st.session_state.last_retrieved_docs = docs

                    docs = rank_docs(docs, prompt)
                    docs = [d for d in docs if d.metadata.get("source") not in st.session_state.distrusted_sources]

                    doc_model = choose_model(prompt, st.session_state.doc_model)
                    if "flowchart" in prompt.lower():
                        chain = build_flowchart_answer_chain(doc_model)
                    elif any(w in prompt.lower() for w in ["how to", "steps", "instructions"]):
                        chain = build_steps_answer_chain(doc_model)
                    elif any(w in prompt.lower() for w in ["what is", "who is", "when was", "list"]):
                        chain = build_single_answer_chain(doc_model)
                    else:
                        chain = build_multi_answer_chain(doc_model)

                    if st.session_state.answer_display_mode == "Ranked Answers":
                        for i, doc in enumerate(docs[:st.session_state.k_retrieval], start=1):
                            context = doc.page_content
                            answer_i = safe_invoke(chain, {"context": context, "question": prompt})
                            rendered_answers.append({
                                "rank": i,
                                "text": answer_i,
                                "source_type": "document",
                                "source": doc.metadata.get("source", "unknown"),
                                "chunk_id": doc.metadata.get("chunk_id"),
                                "content": doc.page_content
                            })
                        # append general answer last
                        rendered_answers.append(general_rendered)
                        for r in rendered_answers:
                            if r["source_type"] == "document" and r.get("rank"):
                                st.markdown(f"**Rank {r['rank']} Answer:**\n{r['text']}")
                            elif r["source_type"] == "general":
                                st.markdown("**General Answer:**\n" + r["text"])
                    else:
                        context = "\n\n".join([d.page_content for d in docs])
                        doc_answer_text = safe_invoke(chain, {"context": context, "question": prompt})
                        rendered_answers.append({
                            "rank": 1,
                            "text": doc_answer_text,
                            "source_type": "document",
                            "source": ", ".join(sorted({d.metadata.get("source", "unknown") for d in docs})),
                            "chunk_id": None,
                            "content": context
                        })
                        rendered_answers.append(general_rendered)
                        st.markdown(rendered_answers[0]["text"])
                else:
                    rendered_answers = [general_rendered]
                    st.markdown(general_answer)

            # Persist rendered_answers into the session message so it survives reruns
            if st.session_state.current_chat_session.get("messages"):
                st.session_state.current_chat_session["messages"][-1]["answer"] = (
                    "\n\n".join([f"Rank {r['rank']}: {r['text']}" if r.get('rank') else f"General: {r['text']}" for r in rendered_answers])
                    if rendered_answers else ""
                )
                st.session_state.current_chat_session["messages"][-1]["rendered_answers"] = rendered_answers

            # Save session and rerun to ensure buttons and viewer work on stable session state
            save_chat_session(st.session_state.current_chat_session)
            st.rerun()

    # If a source was opened (user clicked a View source button), render the viewer below chat area
    if st.session_state.get("open_source"):
        opened = st.session_state.open_source
        st.markdown("---")
        st.markdown("### Source Viewer")
        left_col, right_col = st.columns([1, 2])
        with left_col:
            st.markdown("**Answers (collapsed)**")
            # show brief previews of last message's rendered answers
            last_msg = st.session_state.current_chat_session["messages"][-1] if st.session_state.current_chat_session.get("messages") else None
            if last_msg and last_msg.get("rendered_answers"):
                for r in last_msg["rendered_answers"]:
                    short_preview = (r["text"][:300] + "...") if len(r["text"]) > 300 else r["text"]
                    if r.get("rank"):
                        st.markdown(f"**Rank {r['rank']}** — {short_preview}")
                    else:
                        st.markdown(f"**General** — {short_preview}")
        with right_col:
            if opened["source_type"] == "document":
                st.markdown(f"**Document:** `{opened['source']}`")
                st.markdown(f"**Chunk ID:** `{opened.get('chunk_id')}`")
                content_html = _highlight_query_terms(opened.get("content", ""), opened.get("highlight_query", ""))
                st.markdown(content_html, unsafe_allow_html=True)
                surrounding = [d for d in st.session_state.last_retrieved_docs if d.metadata.get("source") == opened["source"]]
                if surrounding:
                    st.markdown("**Other chunks from this source (preview):**")
                    for s in surrounding[:5]:
                        preview = s.page_content
                        preview_short = (preview[:300] + "...") if len(preview) > 300 else preview
                        st.markdown(f"- `{s.metadata.get('chunk_id')}` — {preview_short}")
                if st.button("❌ Distrust this source", key=f"{opened['source']}_distrust_btn"):
                    st.session_state.distrusted_sources.add(opened["source"])
                    st.info(f"Source `{opened['source']}` added to distrust list.")
                    st.session_state.open_source = None
                    st.rerun()
                if isinstance(opened["source"], str) and opened["source"].lower().startswith("http"):
                    st.markdown(f"[Open original URL ▶]({opened['source']})")
                else:
                    st.markdown("_Local/Uploaded document (filename shown). If you keep original files accessible, we can add a direct open-link._")
            else:
                st.markdown(f"**General model source:** `{opened.get('source')}`")
                st.markdown("This answer was produced by the general chat model, not extracted from a stored document.")
                if isinstance(opened.get("source"), str) and opened.get("source").lower().startswith("http"):
                    st.markdown(f"[Open source URL ▶]({opened.get('source')})")


def new_chat():
    new_session = create_new_chat_session()
    st.session_state.all_chat_sessions.insert(0, new_session)
    st.session_state.current_chat_session = new_session
    st.rerun()


def delete_chat(session_id):
    delete_chat_session(session_id)
    delete_vectorstore_for_session(session_id)
    st.session_state.all_chat_sessions = [s for s in st.session_state.all_chat_sessions if s["id"] != session_id]
    if not st.session_state.all_chat_sessions:
        new_chat()
    elif st.session_state.current_chat_session["id"] == session_id:
        st.session_state.current_chat_session = st.session_state.all_chat_sessions[0]
    st.rerun()


if check_authentication():
    run_app()
else:
    st.info("Please enter a valid App API key in the sidebar.")
    st.stop()
