
const N = 24 * 2
export const threshold = Math.random() * 1
export const variables = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

// {
//     "total":{"n":4234,"percent":23},
//     "variables":[
//         {"name":"A","n":343,"percent":13},
//         {"name":"B","n":434,"percent":20},
//         ...
//     ]
// }

export const stats = {
    total: { n: 4234, percent: 23 },
    variables: [
        { name: 'A', n: 343, percent: 13 },
        { name: 'B', n: 434, percent: 20 },
        { name: 'C', n: 1, percent: 0 },
        { name: 'D', n: 55, percent: -7 },
        { name: 'E', n: 17, percent: 1 },
        { name: 'F', n: 0, percent: -100 },
        { name: 'G', n: 3, percent: -13 },
    ]
}


export const variable_chart = [
    { name: 'A', value: 60 },
    { name: 'B', value: 30 },
    { name: 'C', value: 10 }
]

// [
//     {"time":"0:00","value":18},
//     {"time":"1:00","value":19},
//     {"time":"2:00","value":20},
//     {"time":"3:00","value":2},
//     ...
// ]


export let hourly_chart = []
for (let index = 0; index < 24; index++) {
    hourly_chart.push({ time: index.toString() + ':00', value: Math.floor(Math.random() * 24) })
}

// [
//     {"time":"Monday","value":83},
//     {"time":"Tuesday","value":37},
//     {"time":"Wednesday","value":6},
//     {"time":"Thursday","value":84},
//     ...
// ]


export let weekly_chart = []
for (const day of ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']) {
    weekly_chart.push({ time: day, value: Math.floor(Math.random() * 100) })
}

// [
//     {"date":"2020-02-02 02:02:02","value":335,"name":"A","score":0.4173808469145217,"label":1},
//     {"date":"2020-02-02 02:02:02","value":62,"name":"B","score":0.846660483478745,"label":1},
//     {"date":"2020-02-02 02:02:02","value":875,"name":"C","score":0.07021022554222522,"label":1},
//     {"date":"2020-02-02 02:02:02","value":616,"name":"D","score":0.7063871943877669,"label":1},
//     {"date":"2020-02-02 02:02:02","value":364,"name":"E","score":0.3072443491503909,"label":1},
//     ...
// ]

export let main_chart = []
for (let index = 0; index < N; index++) {
    let date = moment('2021-01-01 00:00:00').add(index * 60, 'minutes').format('YYYY-MM-DD HH:mm:ss')
    for (const variable of variables) {
        let score = Math.random()
        main_chart.push({
            date: date,
            value: Math.floor(Math.random() * 1000),
            name: variable,
            score: score,
            label: Number(score > threshold),
        })
    }
}

// [
//     {"date":"2020-02-02 02:02:02","value":54,"name":"A","score":0.6782596376507299,"label":1},
//     {"date":"2020-02-02 02:02:02","value":170,"name":"B","score":0.14900607167777014,"label":0},
//     {"date":"2020-02-02 02:02:02","value":277,"name":"C","score":0.9234196997734485,"label":1},
//     {"date":"2020-02-02 02:02:02","value":948,"name":"D","score":0.7746902020612063,"label":1},
//     {"date":"2020-02-02 02:02:02","value":666,"name":"E","score":0.01617696163633453,"label":0},
//     ...
// ]

export const simulate_data = (n_data, last_date) => {
    let new_data = []
    for (let index = 1; index < n_data + 1; index++) {
        // let date = moment(last_date).add(index * 60, 'minutes').format('YYYY-MM-DD HH:mm:ss')
        let date = '2020-02-02 02:02:02'
        for (const variable of variables) {
            let score = Math.random()
            new_data.push({
                date: date,
                value: Math.floor(Math.random() * 1000),
                name: variable,
                score: score,
                label: Number(score > threshold),
            })
        }
    }
    return new_data
}

// [
//     {"date":"2020-02-02 02:02:02","score":0.515492833207886},
//     {"date":"2020-02-02 02:02:02","score":0.515492833207886},
//     {"date":"2020-02-02 02:02:02","score":0.515492833207886},
//     {"date":"2020-02-02 02:02:02","score":0.515492833207886},
//     ...
// ]

export let anomaly_score_chart = []
for (let index = 0; index < N; index++) {
    // let date = moment('2021-01-01 00:00:00').add(index * 60, 'minutes').format('YYYY-MM-DD HH:mm:ss')
    let date = '2020-02-02 02:02:02'
    let item_date = main_chart.filter(item => item.date === date)
    let sum_scores = 0
    item_date.forEach(item => {
        sum_scores += item.score
    });
    let avg_score = sum_scores / item_date.length
    anomaly_score_chart.push({
        date: date,
        score: avg_score,
    })
}

// [
//     {"date":"1","range":"anomaly","name":"A","value":28.396776227415856,"label":0},
//     {"date":"2","range":"anomaly","name":"A","value":95.08927683043797,"label":0},
//     {"date":"3","range":"anomaly","name":"A","value":56.84303242511719,"label":0},
//     {"date":"4","range":"anomaly","name":"A","value":91.58144633328453,"label":0},
//     {"date":"5","range":"anomaly","name":"A","value":34.604360320391514,"label":0},
//     ...
// ]

export const close_pattern_chart = []
for (const variable of variables) {
    for (const range of ['anomaly', '21.05.26 - 21.06.26', '21.04.26 - 21.05.26', '21.03.26 - 21.04.26', '21.02.26 - 21.03.26']) {
        for (let index = 1; index < 31; index++) {
            if (range === 'anomaly') {
                close_pattern_chart.push({
                    date: index.toString(),
                    range: range,
                    name: variable,
                    value: index < 10 || index > 20 ? Math.random() * 100 : 100 + Math.random() * 500,
                    label: index < 10 || index > 20 ? 0 : 1
                })
            } else {
                close_pattern_chart.push({
                    date: index.toString(),
                    range: range,
                    name: variable,
                    value: Math.random() * 100,
                    label: 0
                })
            }
        }
    }
}

// [
//     {"name":"A","value":672.3315208812917},
//     {"name":"A","value":-135.52764778981103},
//     {"name":"A","value":429.92692413810454},
//     {"name":"A","value":-105.60365359791449},
//     ...
// ]

export const possible_outliers = []
for (const variable of variables) {
    for (let index = 0; index < 10; index++) {
        possible_outliers.push({
            name: variable,
            value: index % 2 === 0 ? Math.random() * 1000 : Math.random() * -200
        })

    }
}

// [
//     {"date":"2020-02-02 0:00:00","score":null,"name":"Overall"},
//     {"date":"2020-02-02 1:00:00","score":null,"name":"Overall"},
//     {"date":"2020-02-02 2:00:00","score":null,"name":"Overall"},
//     {"date":"2020-02-02 3:00:00","score":null,"name":"Overall"},
//     ...
// ]

export let score_heatmap = []
for (let index = 0; index < N; index++) {
    // let date = moment('2021-01-01 00:00:00').add(index * 60, 'minutes').format('YYYY-MM-DD HH:mm:ss')
    let date = '2020-02-02 02:02:02'
    let rev_vars = variables.slice().reverse()
    rev_vars.push('Overall')
    for (const variable of rev_vars) {
        if (variable === 'Overall') {
            let date_data = main_chart.filter(data => data.date === date)
            let sum_score = 0
            date_data.forEach(item => {
                sum_score += item.score
            })
            let daily_score = sum_score / date_data.length
            score_heatmap.push({
                date: date,
                score: daily_score,
                name: variable
            })
        } else {
            score_heatmap = score_heatmap.concat(main_chart.filter(data => data.date === date && data.name == variable))
        }
    }

}