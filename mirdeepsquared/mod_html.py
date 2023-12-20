from bs4 import BeautifulSoup as bs
import sys
import argparse


def remove_false_positives(html_file, true_positives):
    with open(html_file) as html:
        soup = bs(html, 'html.parser')
        novel_items = soup.find(string="novel miRNAs predicted by miRDeep2").findNext('table')
        novel_item = novel_items.findNext('tr')
        while novel_item is not None:
            candidate = novel_item.findNext('td').contents[0].text
            next_novel_item = novel_item.findNext('tr')
            if candidate not in true_positives:
                novel_item.decompose()
            if next_novel_item.parent is not novel_items:
                # Don't remove items from the mature section
                break
            novel_item = next_novel_item

        new_header = bs('miRDeep<sup>2</sup>', 'html.parser')
        # TODO: also add mirdeep^2 version / models used information, perhaps just resort the novel section according to the prediction confidence?
        # Perhaps with https://www.cssscript.com/fast-html-table-sorting/? This would basically require a whole new result.html as the table headers already are "clickable"
        soup.find('b', string="miRDeep2").replace_with(new_header)
    return soup.prettify("utf-8")


def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-html', description='Only keeps the provided true positives in the novel section in mirdeeps result html')

    parser.add_argument('html_file', help="Path to the html file with too many results")  # positional argument
    parser.add_argument('output', help="Where to put the filtered html")  # positional argument
    parser.add_argument('true_positives', nargs='?', help='Input file with true positives in it, if empty stdin is used', type=argparse.FileType('r'), default=sys.stdin)

    parsed_args = parser.parse_args(args)
    if parsed_args.true_positives.isatty():
        parser.print_help()
        sys.exit(1)
    return parsed_args


def main():
    args = parse_args(sys.argv[1:])
    # args = parse_args(['tests/example_mirdeep_output/zebrafish_result_13_11_2023_t_18_47_00.html', 'tmp.html', "--", "tp.txt"])
    true_positives = set([true_positive.rstrip('\n') for true_positive in args.true_positives.readlines()])

    html_without_false_positives = remove_false_positives(args.html_file, true_positives)
    with open(args.output, "wb") as f_output:
        f_output.write(html_without_false_positives)


if __name__ == '__main__':
    main()
