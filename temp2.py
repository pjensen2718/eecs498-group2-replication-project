import itertools

def gen_user_prompt_2s() -> list[str]:
    indices = [0, 1, 2, 3]
    metrics_phrases = ['grammar', 'creativity', "consistency with the story's beginning", 'whether the plot makes sense']
    metrics = ['Grammar', 'Creativity', 'Consistency', 'Plot']
    idx_perms = list(itertools.permutations(indices))
    
    user_prompt_2_perms = []
    for i in idx_perms:
        user_prompt_2 = f"""
        Now, grade the student's completion in terms of {metrics_phrases[i[0]]}, {metrics_phrases[i[1]]}, 
        {metrics_phrases[i[2]]} and {metrics_phrases[i[3]]}. The scores for each of these categories should be an integer out of 10. 
        Moreover, please provide your best guess of what the age of the student 
        might be, as reflected from the completion. Choose from possible age 
        groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16.

        Format your output as follows:
        {metrics[i[0]]}: X/10, {metrics[i[1]]}: X/10, {metrics[i[2]]}: X/10, {metrics[i[3]]}: X/10, Age group: X (Y-Z)
        """
        user_prompt_2_perms.append(user_prompt_2)

    return user_prompt_2_perms
        
    

def swap_grade_locations(story_completion: str) -> tuple[str, list[str]]:
    # do we want to do every combination (24) of the ordering of grades?
    user_prompt_1 = f"""
    Your task is to evaluate the performance of a student. The student is 
    given the following exercise. In the following exercise, the student is 
    given a beginning of a story. The student needs to complete it into a full 
    story. The exercise tests the student's language abilities and creativity. 
    The symbol *** marks the separator between the prescribed beginning and 
    the student's completion:
    
    The student wrote the following story:

    {story_completion}

    Please provide your general assessment about the story written by the 
    student (the one after the *** symbol). Please be concise. Is it 
    gramatically correct? Is it consistent with the requirements in the 
    exercise? Is it consistent with the beginning of the story? Pay special 
    attention to whether the student manages to complete the sentence which is 
    split in the middle by the separator ***.
    """

    user_prompt_2s = []
    user_prompt_2s = gen_user_prompt_2s()

    return user_prompt_1, user_prompt_2s

output = swap_grade_locations("Hello, world!")
if len(output[1]) != 24: print("ERROR: 24 permutations not created")
print(output[0] + "\n---")
for prompt_2 in output[1]:
    print(prompt_2)