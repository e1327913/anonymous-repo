import json_repair

def step(inputs, model, tokenizer, max_generated_tokens, temperature):
    # Text Generation
    outputs = model.generate(**inputs, max_new_tokens=max_generated_tokens, do_sample=True, temperature=temperature)
    # Decode Outputs
    outputs = tokenizer.batch_decode(outputs)
    return outputs

def get_json_chunk(item):
    new_history = []
    chunks = item.replace('<|begin_of_text|>', '').replace('<|end_of_text|>', '').split('<|eot_id|>')
    for chunk in chunks:
        details = [detail for detail in chunk.split('\n\n')]
        role = details[0].replace('<|start_header_id|>', '').replace('<|end_header_id|>', '').strip()
        content = ' '.join(details[1:]).strip()
        content = json_repair.loads(content)
        if len(role) > 0 and len(content) > 0:
            new_history.append({'role': role, 'content': content})
    return new_history

def format_chunks(item):
    new_history = []
    chunks = item.replace('<|begin_of_text|>', '').replace('<|end_of_text|>', '').split('<|eot_id|>')
    for chunk in chunks:
        details = [detail for detail in chunk.split('\n\n')]
        role = details[0].replace('<|start_header_id|>', '').replace('<|end_header_id|>', '').strip()
        content = ' '.join(details[1:]).strip()
        if len(role) > 0 and len(content) > 0:
            new_history.append({'role': role, 'content': content})
    return new_history