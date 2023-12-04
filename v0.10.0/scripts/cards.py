"""
Adapted from pymdownx.tabbed (https://github.com/facelessuser/pymdown-extensions/)
"""
import re
import xml.etree.ElementTree as etree

from markdown import Extension
from markdown.blockprocessors import BlockProcessor
from markdown.extensions.attr_list import AttrListTreeprocessor, get_attrs


def assign_attrs(elem, attrs):
    """Assign `attrs` to element."""
    for k, v in get_attrs(attrs):
        if k == ".":
            # add to class
            cls = elem.get("class")
            if cls:
                elem.set("class", "{} {}".format(cls, v))
            else:
                elem.set("class", v)
        else:
            # assign attribute `k` with `v`
            elem.set(AttrListTreeprocessor.NAME_RE.sub("_", k), v)


class CardProcessor(BlockProcessor):
    """card block processor."""

    START = re.compile(r"(?:^|\n)={3} *(card)?(?: +({:.*?}) *(?:\n|$))?")
    COMPRESS_SPACES = re.compile(r" {2,}")

    def __init__(self, parser, config):
        """Initialize."""

        super().__init__(parser)
        self.card_group_count = 0
        self.current_sibling = None
        self.content_indention = 0

    def detab_by_length(self, text, length):
        """Remove a card from the front of each line of the given text."""

        newtext = []
        lines = text.split("\n")
        for line in lines:
            if line.startswith(" " * length):
                newtext.append(line[length:])
            elif not line.strip():
                newtext.append("")  # pragma: no cover
            else:
                break
        return "\n".join(newtext), "\n".join(lines[len(newtext) :])

    def parse_content(self, parent, block):
        """
        Get sibling card.

        Retrieve the appropriate sibling element. This can get tricky when
        dealing with lists.

        """

        old_block = block
        non_cards = ""
        card_set = "card-set"

        # We already acquired the block via test
        if self.current_sibling is not None:
            sibling = self.current_sibling
            block, non_cards = self.detab_by_length(block, self.content_indent)
            self.current_sibling = None
            self.content_indent = 0
            return sibling, block, non_cards

        sibling = self.lastChild(parent)

        if (
            sibling is None
            or sibling.tag.lower() != "div"
            or sibling.attrib.get("class", "") != card_set
        ):
            sibling = None
        else:
            # If the last child is a list and the content is indented sufficient
            # to be under it, then the content's is sibling is in the list.
            last_child = self.lastChild(sibling)
            card_content = "card-content"
            child_class = (
                last_child.attrib.get("class", "") if last_child is not None else ""
            )
            indent = 0
            while last_child is not None:
                if (
                    sibling is not None
                    and block.startswith(" " * self.tab_length * 2)
                    and last_child is not None
                    and (
                        last_child.tag in ("ul", "ol", "dl")
                        or (last_child.tag == "div" and child_class == card_content)
                    )
                ):
                    # Handle nested card content
                    if last_child.tag == "div" and child_class == card_content:
                        temp_child = self.lastChild(last_child)
                        if temp_child is None or temp_child.tag not in (
                            "ul",
                            "ol",
                            "dl",
                        ):
                            break
                        last_child = temp_child

                    # The expectation is that we'll find an `<li>`.
                    # We should get it's last child as well.
                    sibling = self.lastChild(last_child)
                    last_child = (
                        self.lastChild(sibling) if sibling is not None else None
                    )
                    child_class = (
                        last_child.attrib.get("class", "")
                        if last_child is not None
                        else ""
                    )

                    # Context has been lost at this point, so we must adjust the
                    # text's indentation level so it will be evaluated correctly
                    # under the list.
                    block = block[self.tab_length :]
                    indent += self.tab_length
                else:
                    last_child = None

            if not block.startswith(" " * self.tab_length):
                sibling = None

            if sibling is not None:
                indent += self.tab_length
                block, non_cards = self.detab_by_length(old_block, indent)
                self.current_sibling = sibling
                self.content_indent = indent

        return sibling, block, non_cards

    def test(self, parent, block):
        """Test block."""

        if self.START.search(block):
            return True
        else:
            return self.parse_content(parent, block)[0] is not None

    def run(self, parent, blocks):
        """Convert to card block."""

        block = blocks.pop(0)
        m = self.START.search(block)
        card_set = "card-set"

        if m:
            # removes the first line
            if m.start() > 0:
                self.parser.parseBlocks(parent, [block[: m.start()]])
            block = block[m.end() :]
            sibling = self.lastChild(parent)
            block, non_cards = self.detab(block)
        else:
            sibling, block, non_cards = self.parse_content(parent, block)

        if m:
            if (
                sibling is not None
                and sibling.tag.lower() == "div"
                and sibling.attrib.get("class", "") == card_set
            ):
                card_group = sibling
            else:
                self.card_group_count += 1
                card_group = etree.SubElement(
                    parent,
                    "div",
                    {
                        "class": card_set,
                        "data-cards": "%d:0" % self.card_group_count,
                    },
                )

            data = card_group.attrib["data-cards"].split(":")
            card_set = int(data[0])
            card_count = int(data[1]) + 1

            div = etree.SubElement(
                card_group,
                "div",
                {
                    "class": "card-content",
                },
            )
            attributes = m.group(2)

            if attributes:
                attr_m = AttrListTreeprocessor.INLINE_RE.search(attributes)
                if attr_m:
                    assign_attrs(div, attr_m.group(1))
                    if div.get("href"):
                        div.tag = "a"

            card_group.attrib["data-cards"] = "%d:%d" % (card_set, card_count)
        else:
            if sibling.tag in ("li", "dd") and sibling.text:
                # Sibling is a list item, but we need to wrap it's content should be
                # wrapped in <p>
                text = sibling.text
                sibling.text = ""
                p = etree.SubElement(sibling, "p")
                p.text = text
                div = sibling
            elif sibling.tag == "div" and sibling.attrib.get("class", "") == card_set:
                # Get `card-content` under `card-set`
                div = self.lastChild(sibling)
            else:
                # Pass anything else as the parent
                div = sibling

        self.parser.parseChunk(div, block)

        if non_cards:
            # Insert the card content back into blocks
            blocks.insert(0, non_cards)


class CardExtension(Extension):
    """Add card extension."""

    def __init__(self, *args, **kwargs):
        """Initialize."""

        self.config = {
            "slugify": [
                0,
                "Slugify function used to create card specific IDs - Default: None",
            ],
            "combine_header_slug": [
                False,
                "Combine the card slug with the slug of the parent header - "
                "Default: False",
            ],
            "separator": ["-", "Slug separator - Default: '-'"],
        }

        super(CardExtension, self).__init__(*args, **kwargs)

    def extendMarkdown(self, md):
        """Add card to Markdown instance."""
        md.registerExtension(self)

        config = self.getConfigs()

        self.card_processor = CardProcessor(md.parser, config)

        md.parser.blockprocessors.register(
            self.card_processor,
            "card",
            105,
        )

    def reset(self):
        """Reset."""

        self.card_processor.card_group_count = 0


def makeExtension(*args, **kwargs):
    """Return extension."""

    return CardExtension(*args, **kwargs)
