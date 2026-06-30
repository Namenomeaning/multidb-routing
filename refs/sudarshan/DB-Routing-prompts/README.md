# DB-Routing-Prompts

Prompts for routing natural language queries to the correct database in multi-database environments.

## 🎯 About

This repository contains prompts used in our **LLM Re-ranking with Modular Reasoning** approach for database routing. Rather than asking an LLM to directly rank databases, we break the problem into focused sub-tasks for more accurate results.

## 📁 Prompts

### `adjacency_list_prompt.txt`
Generates a join graph between tables in a database schema.

**Input:** Database schema (DDL)  
**Output:** Adjacency list (e.g., `0:1,2,5` means Table 0 can join with Tables 1, 2, 5)

### `phrase_mapping_prompt.txt`
Maps query terms to database columns.

**Input:** Natural language query + Database schema  
**Output:** Mappings (e.g., `John - Student.student_name`)

## 🔧 How It Works

```
Query → Embedding Retrieval (Top-k DBs) → Modular Re-ranking → Final Ranking
```

**Re-ranking Steps:**
1. Build table join graph (once per DB)
2. Map query phrases to schema columns
3. Compute coverage & connectivity scores
4. Break ties using semantic similarity


