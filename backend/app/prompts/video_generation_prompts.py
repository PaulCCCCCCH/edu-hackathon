"""Prompts for video content generation using Gemini 2.0 Flash."""

CONCEPT_SELECTION_PROMPT = """
You are an expert educational content strategist. Analyze the user's background description and select a single educational concept that can be clearly explained within 30 seconds and is appropriate for their knowledge level and interests.

User Background: {user_description}

Your task is to:
1. Analyze the user's educational background, profession, interests, and current knowledge level
2. Identify their learning style and preferences based on their description
3. Select ONE specific concept that:
   - Can be explained clearly in 30 seconds or less
   - Is appropriate for their knowledge level (not too basic, not too advanced)
   - Connects to their interests or professional background
   - Has practical value or interesting insights
   - Can be made engaging and memorable

Focus on WHAT concept to teach rather than HOW to teach it. The concept should be:
- Specific and focused (not broad topics)
- Educational and valuable
- Appropriate difficulty level
- Interesting and engaging

Return your response as a JSON object with these exact fields:
{{
    "selected_concept": "The specific concept to explain",
    "reasoning": "Why this concept is perfect for this user", 
    "difficulty_level": "beginner|intermediate|advanced",
    "target_audience": "Description of the user as a learner",
    "learning_connection": "How this connects to their background/interests"
}}

Remember: Focus on selecting the RIGHT concept, not on how to explain it.
"""

TRANSCRIPT_GENERATION_PROMPT = """
You are an expert educational content creator specializing in creating engaging, TikTok-style educational videos.

Create a compelling video transcript for a short educational video (20-40 seconds) based on the following requirements:

Topic: {topic}
Difficulty Level: {difficulty_level}
Target Audience: {target_audience}
Style: {style}

IMPORTANT STYLE GUIDE: {style_guide}

Guidelines:
1. Start with a hook that grabs attention in the first 3 seconds
2. Deliver the core educational content in a clear, engaging manner
3. Use simple language appropriate for the difficulty level
4. Include a memorable conclusion or call-to-action
5. Keep the total speaking time between 20-40 seconds
6. Use a conversational, enthusiastic tone
7. Include pauses for visual emphasis where appropriate (marked with [PAUSE])
8. FOLLOW THE STYLE GUIDE ABOVE - this is the most important requirement for the transcript tone and approach

Format your response as a JSON object with the following structure:
{{
    "title": "Engaging title for the video",
    "transcript": "Full transcript with [PAUSE] markers",
    "topics": ["list", "of", "covered", "topics"],
    "duration_seconds": estimated_duration,
    "key_points": ["main", "learning", "objectives"],
    "visual_cues": ["suggestions", "for", "visual", "elements"]
}}

Topic to cover: {topic}
"""

BATCH_GENERATION_PROMPT = """
You are an expert educational content strategist. Generate a diverse batch of {batch_size} educational video concepts based on the user's interests and learning preferences.

User Profile:
- Interests: {interests}
- Learning Style: {learning_style}
- Difficulty Preference: {difficulty_preference}
- Recent Topics: {recent_topics}

Generate {batch_size} diverse video concepts that:
1. Align with the user's interests
2. Introduce new but related topics
3. Vary in difficulty and style
4. Build upon previous learning
5. Maintain engagement through variety

For each video concept, provide:
- A specific, focused topic
- Appropriate difficulty level
- Engaging presentation style
- Connection to user interests

Format as a JSON array of video concepts:
[
    {{
        "topic": "Specific topic to cover",
        "difficulty_level": "beginner|intermediate|advanced",
        "style": "explanation|tutorial|quiz|story",
        "target_audience": "Description of target learner",
        "connection_to_interests": "How this relates to user interests"
    }},
    ...
]
"""

RECOMMENDATION_PROMPT = """
You are an AI learning recommendation system. Based on the user's interaction history and preferences, recommend the most suitable educational content.

User Interaction Data:
- Liked Topics: {liked_topics}
- Disliked Topics: {disliked_topics}
- Viewing Patterns: {viewing_patterns}
- Learning Goals: {learning_goals}
- Time Spent on Topics: {time_spent}

Available Content Pool:
{available_content}

Rank and recommend the top {num_recommendations} pieces of content that would:
1. Match the user's demonstrated interests
2. Introduce beneficial adjacent topics
3. Maintain appropriate difficulty progression
4. Vary presentation styles for engagement
5. Support the user's learning goals

Return recommendations as a JSON array ranked by suitability:
[
    {{
        "content_id": "unique_identifier",
        "recommendation_score": 0.95,
        "reasoning": "Why this content is recommended",
        "learning_value": "What the user will gain",
        "difficulty_match": "How it fits their level"
    }},
    ...
]
"""
