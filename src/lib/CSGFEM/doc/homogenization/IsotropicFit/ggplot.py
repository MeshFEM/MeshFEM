from csv import DictReader
from mako.template import Template
from mako.runtime import Context
from subprocess import check_call

class ggplot:
    def __init__(self):
        self.template = Template(filename="plot.mako");
        self.r_script = "";

    def set_data(self, csv_file):
        self.r_script += self.template.get_def("header").render(
                csv_file=csv_file);
        self.__load_csv_file(csv_file);

    def filter_data(self, col_name, min_val, max_val):
        self.r_script += self.template.get_def("filter_data").render(
                field_name = col_name,
                lower_bound = min_val,
                upper_bound = max_val);

    def add_column(self, col_name, formula):
        self.r_script += self.template.get_def("add_column").render(
                new_column = col_name, formula = formula);

    def discretize_column(self, col_name):
        self.r_script += self.template.get_def("discretize_column").render(
                field_name = col_name);

    def scatter_plot(self, x_col, y_col, w_col, discrete=False):
        self.r_script += self.template.get_def("scatter_plot").render(
                x_col = x_col, y_col = y_col, w_col = w_col, discrete=discrete);

    def title(self, title_text):
        self.r_script += self.template.get_def("title").render(
                title_text = title_text);

    def histogram(self, w_col, num_bins=20):
        self.r_script += self.template.get_def("histogram").render(
                w_col = w_col, num_bins = num_bins);

    def set_range(self, x_min, x_max, y_min, y_max):
        self.r_script += self.template.get_def("set_range").render(
                x_col_min = x_min, x_col_max = x_max,
                y_col_min = y_min, y_col_max = y_max);

    def facet_grid(self, facet_1, facet_2):
        self.r_script += self.template.get_def("facet_grid").render(
                facet_1 = facet_1, facet_2 = facet_2);

    def save_plot(self, out_name, width=10, height=6):
        self.r_script += self.template.get_def("save_plot").render(
                out_name = out_name, width = width, height = height);

    def draw_point(self, x, y, color="green", size=3):
        self.r_script += self.template.get_def("draw_point").render(
                x=x, y=y, color=color, size=size);

    def draw_text(self, x, y, text, hjust=0, vjust=1):
        self.r_script += self.template.get_def("draw_text").render(
                x=x, y=y, text=text, hjust=hjust, vjust=vjust);

    def plot(self):
        r_file = "plot.r";
        with open(r_file, 'w') as fout:
            fout.write(self.r_script);
        command = "Rscript {}".format(r_file);
        check_call(command.split());

    def get_col(self, col_name):
        return self.data[col_name];

    def __load_csv_file(self, csv_file):
        with open(csv_file, 'r') as fin:
            reader = DictReader(fin);
            self.data = {};
            for field in reader.fieldnames:
                self.data[field] = [];

            for row in reader:
                for key in row:
                    self.data[key].append(row[key]);

