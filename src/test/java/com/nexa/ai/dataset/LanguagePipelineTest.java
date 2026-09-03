package com.nexa.ai.dataset;

import com.nexa.ai.tokenizer.SimpleTokenizer;
import com.nexa.ai.tokenizer.TokenizerTrainer;
import com.nexa.ai.tokenizer.Vocabulary;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class LanguagePipelineTest {

    @Test
    void shouldConvertTextIntoTrainingExamples() {

        List<String> texts = List.of(
                "Java is powerful and fast",
                "Java is popular"
        );

        TokenizerTrainer trainer =
                new TokenizerTrainer();

        Vocabulary vocabulary =
                trainer.train(texts);

        SimpleTokenizer tokenizer =
                new SimpleTokenizer(vocabulary);

        List<Integer> tokenIds =
                tokenizer.encode(
                        "Java is powerful and fast"
                );

        LanguageDatasetBuilder builder =
                new LanguageDatasetBuilder();

        LanguageDataset dataset =
                builder.build(tokenIds, 3);

        assertEquals(
                2,
                dataset.size()
        );

        assertEquals(
                3,
                dataset.get(0).inputTokens().size()
        );

        assertEquals(
                3,
                dataset.get(0).targetTokens().size()
        );
    }
}