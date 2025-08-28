def main_loop(args, line, model, tokenizer, knowledge_loop, response_loop, understanding_loop, u_loop):
    all_history_knowledge, all_history_response, all_history_refined = [], [], []
    
    THRESHOLD_ENTAIL = args.threshold_entailment    # default = 0.8
    MAX_LOOP = args.max_loop

    candidates = []
    main_loop_i = 0
    print(f"main_loop {main_loop_i}")
    question = line['question']
    is_correct = line.get('is_correct', None)
    candidate = line['candidate']

    if "generated_knowledge" in line.keys():
        final_knowledge, history_knowledge = knowledge_loop(args, model, tokenizer, question, [line['generated_knowledge']])
    else:
        final_knowledge, history_knowledge = knowledge_loop(args, model, tokenizer, question)
    all_history_knowledge += history_knowledge

    # response_loop
    final_response, history_response, entailment_score_question = response_loop(args, model, tokenizer, question, candidate, final_knowledge, is_correct)
    all_history_response += history_response

    # understandability loop
    if u_loop:
        final_refined, history_refined = understanding_loop(args, model, tokenizer, question, candidate, final_response)
        all_history_refined += history_refined
        candidates.append([entailment_score_question, final_knowledge, final_response, final_refined])
    else:
        candidates.append([entailment_score_question, final_knowledge, final_response])

    # with open("/data/ydh/nlp/output/main_loop_log.txt", "a", encoding="utf-8") as f:
    #     f.write("=== Main loop: " + str(main_loop_i) + " ===\n")
    #     f.write(all_history_knowledge + "\n")
    #     f.write(all_history_response + "\n")
    #     f.write("==================\n\n\n")

    main_loop_i += 1
    while main_loop_i < MAX_LOOP and entailment_score_question < THRESHOLD_ENTAIL:
        print(f"main_loop {main_loop_i}")
        final_knowledge, history_knowledge = knowledge_loop(args, model, tokenizer, question)
        all_history_knowledge += history_knowledge

        final_response, history_response, entailment_score_question = response_loop(args, model, tokenizer, question, candidate, final_knowledge, is_correct)
        all_history_response += history_response

        if u_loop:
            final_refined, history_refined = understanding_loop(args, model, tokenizer, question, candidate, final_response)
            all_history_refined += history_refined
            candidates.append([entailment_score_question, final_knowledge, final_response, final_refined])
        else:
            candidates.append([entailment_score_question, final_knowledge, final_response])

        # with open("/data/ydh/nlp/output/main_loop_log.txt", "a", encoding="utf-8") as f:
        #     f.write("=== Main loop-inside main loop: " + str(main_loop_i) + " ===\n")
        #     f.write(all_history_knowledge + "\n")
        #     f.write(all_history_response + "\n")
        #     f.write("==================\n\n\n")
        
        main_loop_i += 1

    if u_loop:
        if (MAX_LOOP > 1) and entailment_score_question<THRESHOLD_ENTAIL:
            # still not satisified, highest_score
            candidates.sort()
            final_knowledge, final_response, final_refined = candidates[-1][1:]
        return final_knowledge, final_response, final_refined, all_history_knowledge, all_history_response
    else:
        if (MAX_LOOP > 1) and entailment_score_question<THRESHOLD_ENTAIL:
            # still not satisified, highest_score
            candidates.sort()
            final_knowledge, final_response = candidates[-1][1:]
        return final_knowledge, final_response, all_history_knowledge, all_history_response
