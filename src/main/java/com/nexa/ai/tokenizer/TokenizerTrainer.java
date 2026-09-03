package com.nexa.ai.tokenizer;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TokenizerTrainer {

    public Vocabulary train(List<String> texts) {

        if (texts == null || texts.isEmpty()) {
            throw new IllegalArgumentException(
                    "Training texts cannot be null or empty"
            );
        }

        SimpleTokenizer tokenizer =
                new SimpleTokenizer(new Vocabulary());

        Map<String, Integer> frequencies = new HashMap<>();

        for (String text : texts) {

            if (text == null || text.isBlank()) {
                continue;
            }

            List<String> tokens =
                    tokenizer.tokenize(text);

            for (String token : tokens) {
                frequencies.merge(
                        token,
                        1,
                        Integer::sum
                );
            }
        }

        Vocabulary vocabulary = new Vocabulary();

        frequencies.entrySet()
                .stream()
                .sorted(
                        Comparator
                                .<Map.Entry<String, Integer>>
                                        comparingInt(
                                        Map.Entry::getValue
                                )
                                .reversed()
                                .thenComparing(
                                        Map.Entry::getKey
                                )
                )
                .forEach(entry ->
                        vocabulary.addToken(entry.getKey())
                );

        return vocabulary;
    }
}