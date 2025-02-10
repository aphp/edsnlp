from spacy.tokens import Span

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import BaseComponent
from edsnlp.utils.filter import filter_spans
from edsnlp.utils.span_getters import SpanGetter, get_spans


def rel_date_getter(span: Span):
    key = span._._get_key("date")
    if key in span.doc.user_data:
        return span.doc.user_data[key]
    rel_dates_spans = span._.rel["dates"]
    if not rel_dates_spans:
        return None
    date_span = rel_dates_spans[0]
    return date_span.date


class DatesQualifier(BaseComponent):
    def __init__(
        self,
        name: str = "dates_qualifier",
        nlp: PipelineProtocol = None,
        span_getter: SpanGetter = {"ents": True},
        dates_span_getter: SpanGetter = {
            "dates": True,
            "periods": True,
            "ents": ["date", "period"],
        },
        max_left_word_distance: int = 5,
        max_right_word_distance: int = 5,
        max_left_sent_distance: int = 1,
        max_right_sent_distance: int = 1,
    ):
        """
        The `eds.dates_qualifier` component is a rule-based component aiming at
        qualifying entities with dates. This component is parametrized by the
        maximum number of words and sentences between the entity and the date. A date
        is linked with an entity if it is the closest date to the entity within
        either the left/right word distance, OR the left/right sentence distance.

        Examples
        --------

        ```python
        import edsnlp, edsnlp.pipes as eds

        nlp = edsnlp.blank("en")
        nlp.add_pipe(eds.dates())
        nlp.add_pipe(eds.covid())
        nlp.add_pipe(eds.dates_qualifier())

        doc = nlp("Le patient a été hospitalisé le 12/03/2020 pour Covid 19.")
        for ent in doc.ents:
            print(ent.text, "=>", ent._.rel_date)

        # Out:
        # Covid 19 => 2020-03-12
        ```

        Parameters
        ----------
        name: str
            Name of the component.
        nlp: PipelineProtocol
            Pipeline object.
        span_getter: SpanGetter
            Which spans should we link to dates.
        dates_span_getter: SpanGetter
            Which spans are considered as dates.
        max_left_word_distance: int
            Maximum number of words between the entity and the date on the left.
        max_right_word_distance: int
            Maximum number of words between the entity and the date on the right.
        max_left_sent_distance: int
            Maximum number of sentences between the entity and the date on the left.
        max_right_sent_distance: int
            Maximum number of sentences between the entity and the date on the right.
        """
        super().__init__(
            nlp=nlp,
            name=name,
        )
        self.span_getter = span_getter
        self.dates_span_getter = dates_span_getter
        self.max_left_word_distance = max_left_word_distance
        self.max_right_word_distance = max_right_word_distance
        self.max_left_sent_distance = max_left_sent_distance
        self.max_right_sent_distance = max_right_sent_distance

    def set_extensions(self):
        if not Span.has_extension("rel"):
            Span.set_extension("rel", default={})
        if not Span.has_extension("rel_date"):
            Span.set_extension("rel_date", getter=rel_date_getter)

    def __call__(self, doc):
        ents = sorted(get_spans(doc, self.span_getter))
        dates = filter_spans(get_spans(doc, self.dates_span_getter))

        for ent in ents:
            left_date = None
            right_date = None
            left_word_dist = None
            right_word_dist = None
            left_sent_dist = None
            right_sent_dist = None

            for date in reversed(dates):
                if date.end <= ent.start:
                    # ... [ent] ... [right_date] ...
                    right_word_dist = date.start - ent.end
                    if right_word_dist > self.max_right_word_distance:
                        continue
                    right_date = date
                    inter = doc[ent.end : right_date.start]
                    right_sent_dist = sum(1 for w in inter if w.is_sent_start)
                    if right_sent_dist > self.max_right_sent_distance:
                        continue
                    break

            for date in dates:
                if date.start >= ent.end:
                    # ... [left_date] ... [ent] ...
                    left_word_dist = ent.start - date.end
                    if left_word_dist > self.max_left_word_distance:
                        continue
                    left_date = date
                    inter = doc[ent.start : ent]
                    left_sent_dist = sum(1 for w in inter if w.is_sent_start)
                    if left_sent_dist > self.max_left_sent_distance:
                        continue
                    break

            if left_date is not None and right_date is not None:
                if left_sent_dist < right_sent_dist:
                    date = left_date
                elif right_sent_dist < left_sent_dist:
                    date = right_date
                elif left_word_dist < right_word_dist:
                    date = left_date
                elif right_word_dist < left_word_dist:
                    date = right_date
                else:
                    continue
            elif left_date is not None:
                date = left_date
            elif right_date is not None:
                date = right_date
            else:
                continue

            ent._.rel["dates"] = [date]

        return doc
