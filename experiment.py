#!/usr/bin/env python3

from generate_indian_music import generate_music

def experiment_with_prompts():
    # Test prompts with different styles and instruments
    test_prompts = [
        # Classical
        "Peaceful morning raga Bhairav with tanpura drone, slow tempo alap style",
        "Energetic tabla solo in teental, with complex rhythmic patterns",
        
        # Fusion
        "Indian classical fusion with electronic beats, sitar lead and ambient pads",
        "Modern Indian lofi beats with tabla and flute melody, relaxing",
        
        # Folk-inspired
        "Rajasthani folk music with dholak and harmonium, upbeat desert theme",
        "Bengali folk inspired melody with esraj and tabla, emotional and deep",
        
        # Contemporary
        "Modern Indian pop beat with tabla breaks and electronic drums",
        "Indian trap beat with sitar samples and heavy bass, urban style"
    ]
    
    # Generate music for each prompt
    for i, prompt in enumerate(test_prompts):
        print(f"\nExperiment {i+1}:")
        print(f"Prompt: {prompt}")
        output_path = f"experiment_{i+1}.wav"
        generate_music(
            prompt, 
            duration=15,  # 15 seconds duration
            output_path=output_path
        )
        print("-" * 50)

if __name__ == "__main__":
    experiment_with_prompts() 