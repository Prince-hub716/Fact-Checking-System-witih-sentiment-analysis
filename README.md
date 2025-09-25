# Real-Time Fact Checker

## Overview

This is a real-time misinformation detection and fact-checking platform built with Streamlit that combines streaming data processing with AI-powered verification. The system uses Pathway for real-time data streaming, OpenAI's GPT models for intelligent claim analysis, and RAG (Retrieval-Augmented Generation) techniques to verify claims against a curated database of authoritative sources. The platform provides both interactive fact-checking capabilities and live monitoring of claims as they flow through the system.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Multi-page interface with sidebar navigation
- **Interactive Dashboard**: Real-time visualization using Plotly for charts and graphs
- **Session State Management**: Persistent state across user interactions for fact checker instances and processed claims
- **Responsive Layout**: Wide layout configuration optimized for dashboard-style presentation

### Backend Architecture
- **Modular Design**: Separated concerns across three main modules:
  - `fact_checker.py`: Core AI-powered verification logic
  - `pathway_pipeline.py`: Real-time streaming data processing
  - `utils.py`: Shared utility functions and formatting helpers
- **Event-Driven Processing**: Real-time claim processing through Pathway streaming pipeline
- **RAG-Based Verification**: Combines retrieval from authoritative sources with generative AI analysis

### AI and Machine Learning Components
- **OpenAI Integration**: Uses GPT models for intelligent claim analysis and verification
- **Embedding-Based Similarity**: Vector search capabilities for matching claims against known facts
- **Confidence Scoring**: Probabilistic assessment of verification results
- **Multi-Modal Analysis**: Structured prompts for different types of verification tasks

### Data Processing Pipeline
- **Pathway Streaming**: Real-time data processing framework for handling incoming claims
- **Text Preprocessing**: Claim normalization and preparation for analysis
- **Schema Definition**: Structured data flow with defined claim properties (ID, text, timestamp, processing status)
- **Batch and Stream Processing**: Handles both individual fact-check requests and continuous monitoring

### Data Storage and Sources
- **JSON-Based Fact Database**: Structured storage of authoritative sources and verified claims
- **Credibility Scoring**: Source reliability assessment with numerical scores
- **Domain Classification**: Organized by subject areas (health, climate, etc.)
- **Evidence Tracking**: Detailed provenance and supporting evidence for each verification

## External Dependencies

### AI and Machine Learning Services
- **OpenAI API**: GPT model access for natural language processing and claim verification
- **Pathway Framework**: Real-time streaming data processing and transformation engine

### Web Framework and UI
- **Streamlit**: Web application framework for the user interface
- **Plotly**: Interactive charting and visualization library for dashboards and analytics

### Data Processing Libraries
- **Pandas**: Data manipulation and analysis for claim processing
- **NumPy**: Numerical computing support for embeddings and similarity calculations

### Development and Runtime
- **Python Standard Library**: Core functionality including datetime, json, threading, and logging
- **Environment Variables**: Configuration management for API keys and sensitive settings

The architecture prioritizes real-time processing capabilities while maintaining accuracy through AI-powered verification against authoritative sources. The system is designed to scale horizontally through Pathway's streaming architecture and can handle both interactive user requests and continuous background monitoring of data streams.
