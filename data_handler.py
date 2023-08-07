continual_train = {
  'gigaword': {
      'train': [
        ('custom', 'constrain_start+make_a_title'),
        ('custom', 'constrain_contain+make_a_title'),
        ('custom', 'constrain_end+make_a_title'),
      ],
      'test': [
        ('custom', 'constrain_start+make_a_title'),
        ('custom', 'constrain_contain+make_a_title'),
        ('custom', 'constrain_end+make_a_title'),
      ]
  },
  'wiki_auto': {
      'train': [
        ('custom', 'simplification_1'),
      ],
      'test': [
        ('custom', 'simplification_1'),
      ]
  },
  'eli5': {
      'train': [
         ('custom', 'generate_a_question_1'),
        ('custom', 'generate_a_question_2'),
      ],
      'test': [
        ('custom', 'generate_a_question_1'),
      ]
  },
  'empathetic_dialogues': {
      'train': [
        ('custom', "dialogue_with_emotion"),
      ],
      'test':[
         ('custom', "dialogue_with_emotion"),
      ]
  },
  'eSNLI': {
      'train': [
        ('custom', "explain_why"),
      ],
      'test': [
        ('custom', 'explain_why'),
      ]
  },

}
